from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from microring import (
    ALLOWED_PORTS,
    PARAM_COLUMNS,
    add_records_from_path_specs,
    add_records_from_upload_specs,
    add_record_from_paths,
    add_record_from_uploads,
    default_experiment_id,
    ensure_data_layout,
    load_records,
    load_spectra,
    load_spectrum,
    save_records,
    soft_delete_records,
)
from microring.analysis import FitParams, detect_resonances, fit_spectrum, save_fit_outputs
from microring.config import ALLOW_LOCAL_FILE_IMPORT
from microring.data_store import experiment_dir, now_iso, relative_to_project, slug
from microring.plotting import plot_detection, plot_individual, plot_overlap, save_figure
from microring.spectra import SpectraCollection, single_bus_ratio_audit


st.set_page_config(page_title="Microring Analysis", layout="wide")
ensure_data_layout()


def row_label(row: pd.Series) -> str:
    return (
        f"{row.name:02d} | {row.get('record_id')} | {row.get('date')} | "
        f"{row.get('device')} | {row.get('subdevice')} | {row.get('port')} | {row.get('label')}"
    )


def filtered_records(active_only: bool = True) -> pd.DataFrame:
    records = load_records(active_only=active_only)
    with st.expander("Filters", expanded=True):
        cols = st.columns(5)
        date = cols[0].multiselect("Date", sorted(records["date"].dropna().unique()))
        batch = cols[1].multiselect("Batch", sorted(records["batch_name"].dropna().unique()))
        device = cols[2].multiselect("Device", sorted(records["device"].dropna().unique()))
        subdevice = cols[3].multiselect("Subdevice", sorted(records["subdevice"].dropna().unique()))
        port = cols[4].multiselect("Port", sorted(records["port"].dropna().unique()))
    df = records.copy()
    if date:
        df = df[df["date"].isin(date)]
    if batch:
        df = df[df["batch_name"].isin(batch)]
    if device:
        df = df[df["device"].isin(device)]
    if subdevice:
        df = df[df["subdevice"].isin(subdevice)]
    if port:
        df = df[df["port"].isin(port)]
    return df.reset_index(drop=True)


def batch_default() -> str:
    records = load_records(active_only=False)
    if not records.empty and {"batch_name", "date"}.issubset(records.columns):
        dates = set(records["date"].astype(str))
        for value in reversed(records["batch_name"].astype(str).tolist()):
            if value and value not in dates and value.lower() not in {"nan", "batch"}:
                return value
    return "2026_Vernier_Batch1_Row4"


def experiment_inputs(prefix: str):
    c1, c2 = st.columns(2)
    date = c1.text_input("Date", value=pd.Timestamp.today().strftime("%Y_%m_%d"), key=f"{prefix}_date")
    batch_name = c2.text_input("Batch name", value=batch_default(), key=f"{prefix}_batch")
    use_date_folder = st.checkbox("Use date as experiment folder", value=True, key=f"{prefix}_use_date_folder")
    suggested_id = default_experiment_id(date, batch_name)
    if use_date_folder:
        experiment_id = suggested_id
        st.caption(f"Output folder: `experiments/{experiment_id}/`")
    else:
        experiment_id = st.text_input("Experiment folder name", value=suggested_id, key=f"{prefix}_experiment_id")
    return date, batch_name, experiment_id


def parameter_inputs(prefix: str) -> dict:
    st.markdown("**Optional shared parameters**")
    param_cols = st.columns(4)
    params = {}
    for i, col in enumerate(PARAM_COLUMNS):
        value = param_cols[i % 4].text_input(col, value="", key=f"{prefix}_{col}")
        if value:
            params[col] = value
    extra_name = st.text_input("Add custom param column name (without param_ prefix)", value="", key=f"{prefix}_extra_name")
    extra_value = st.text_input("Custom param value", value="", key=f"{prefix}_extra_value")
    if extra_name and extra_value:
        params[f"param_{extra_name.strip()}"] = extra_value
    return params


def tek_number(name: str) -> int | None:
    match = re.search(r"TEK(\d+)", Path(str(name)).name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def sort_key(name: str):
    number = tek_number(name)
    return (number is None, number if number is not None else 0, str(name).lower())


def paired_upload_rows(file_names: list[str], default_device: str, default_subdevice: str, default_port: str, pair_triggers: bool):
    names = sorted(file_names, key=sort_key)
    by_number = {tek_number(name): name for name in names if tek_number(name) is not None}
    used = set()
    rows = []
    for name in names:
        if name in used:
            continue
        trigger_name = ""
        number = tek_number(name)
        if pair_triggers and number is not None:
            candidate = by_number.get(number + 1)
            if candidate and candidate not in used:
                trigger_name = candidate
                used.add(candidate)
        used.add(name)
        rows.append({
            "keep": True,
            "raw_name": name,
            "trigger_name": trigger_name,
            "device": default_device,
            "subdevice": default_subdevice,
            "port": default_port,
            "label": Path(name).stem,
            "notes": "",
        })
    return pd.DataFrame(rows)


def paired_path_rows(paths: list[Path], default_device: str, default_subdevice: str, default_port: str, pair_triggers: bool):
    sorted_paths = sorted(paths, key=lambda p: sort_key(p.name))
    by_number = {tek_number(path.name): path for path in sorted_paths if tek_number(path.name) is not None}
    used = set()
    rows = []
    for path in sorted_paths:
        path_key = str(path)
        if path_key in used:
            continue
        trigger_path = ""
        number = tek_number(path.name)
        if pair_triggers and number is not None:
            candidate = by_number.get(number + 1)
            if candidate is not None and str(candidate) not in used:
                trigger_path = str(candidate)
                used.add(str(candidate))
        used.add(path_key)
        rows.append({
            "keep": True,
            "raw_source_path": str(path),
            "trigger_source_path": trigger_path,
            "device": default_device,
            "subdevice": default_subdevice,
            "port": default_port,
            "label": path.stem,
            "notes": "",
        })
    return pd.DataFrame(rows)


def record_picker(label: str, records: pd.DataFrame, key: str, only_port: str | None = None):
    df = records.copy()
    if only_port is not None:
        df = df[df["port"] == only_port]
    if df.empty:
        st.warning(f"No records available for {label}.")
        return None
    options = {row_label(row): row["record_id"] for _, row in df.iterrows()}
    chosen = st.selectbox(label, list(options), key=key)
    return df[df["record_id"] == options[chosen]].iloc[0]


def database_management():
    st.header("Database Management")
    insert_tab, browse_tab = st.tabs(["Insert Spectrum", "Browse / Edit Records"])

    with insert_tab:
        st.subheader("Insert Spectrum")
        single_tab, bulk_tab = st.tabs(["Single File", "Bulk Insert"])

        with single_tab:
            single_sources = ["Upload CSV files"]
            if ALLOW_LOCAL_FILE_IMPORT:
                single_sources.append("Use local file paths")
            source_mode = st.radio("Source", single_sources, horizontal=True, key="single_source")
            if not ALLOW_LOCAL_FILE_IMPORT:
                st.caption("Local path import is disabled. Set MICRORING_ALLOW_LOCAL_FILE_IMPORT=1 for trusted local use.")
            date, batch_name, experiment_id = experiment_inputs("single")
            c4, c5, c6 = st.columns(3)
            device = c4.text_input("Device", value="V10", key="single_device")
            subdevice = c5.selectbox(
                "Subdevice",
                ["single_bus", "single_ring_1", "single_ring_2", "vernier", "other"],
                key="single_subdevice",
            )
            port = c6.selectbox("Port", ALLOWED_PORTS, key="single_port")
            label = st.text_input("Label", value=f"{device}_{subdevice}_{port}", key="single_label")
            notes = st.text_area("Notes", value="", key="single_notes")
            params = parameter_inputs("single_params")

            if source_mode == "Upload CSV files":
                raw_upload = st.file_uploader("Spectrum/channel CSV", type=["csv"], key="single_raw_upload")
                trigger_upload = st.file_uploader("Trigger CSV (optional)", type=["csv"], key="single_trigger_upload")
                if st.button("Insert uploaded spectrum", type="primary"):
                    if raw_upload is None:
                        st.error("Please upload a spectrum/channel CSV.")
                    else:
                        record = add_record_from_uploads(
                            date=date,
                            batch_name=batch_name,
                            device=device,
                            subdevice=subdevice,
                            port=port,
                            label=label,
                            raw_upload=raw_upload,
                            trigger_upload=trigger_upload,
                            experiment_id=experiment_id,
                            notes=notes,
                            params=params,
                        )
                        st.success(f"Inserted {record['record_id']} into experiments/{slug(experiment_id)}/")
            else:
                raw_path = st.text_input("Spectrum/channel CSV path", key="single_raw_path")
                trigger_path = st.text_input("Trigger CSV path (optional)", key="single_trigger_path")
                if st.button("Copy local files and insert", type="primary"):
                    if not raw_path:
                        st.error("Please provide a spectrum/channel CSV path.")
                    else:
                        record = add_record_from_paths(
                            date=date,
                            batch_name=batch_name,
                            device=device,
                            subdevice=subdevice,
                            port=port,
                            label=label,
                            raw_source_path=raw_path,
                            trigger_source_path=trigger_path or None,
                            experiment_id=experiment_id,
                            notes=notes,
                            params=params,
                        )
                        st.success(f"Inserted {record['record_id']} into experiments/{slug(experiment_id)}/")

        with bulk_tab:
            st.markdown("Add many spectra that share the same date, batch, experiment folder, and optional parameters.")
            bulk_sources = ["Upload many CSV files"]
            if ALLOW_LOCAL_FILE_IMPORT:
                bulk_sources.append("Use local folder")
            bulk_source = st.radio("Bulk source", bulk_sources, horizontal=True, key="bulk_source")
            if not ALLOW_LOCAL_FILE_IMPORT:
                st.caption("Local folder import is disabled. Set MICRORING_ALLOW_LOCAL_FILE_IMPORT=1 for trusted local use.")
            date, batch_name, experiment_id = experiment_inputs("bulk")
            c1, c2, c3 = st.columns(3)
            default_device = c1.text_input("Default device", value="V10", key="bulk_default_device")
            default_subdevice = c2.selectbox(
                "Default subdevice",
                ["single_bus", "single_ring_1", "single_ring_2", "vernier", "other"],
                key="bulk_default_subdevice",
            )
            default_port = c3.selectbox("Default port", ALLOWED_PORTS, index=ALLOWED_PORTS.index("through"), key="bulk_default_port")
            pair_triggers = st.checkbox("Auto-pair TEK channel file with next TEK file as trigger", value=True)
            shared_params = parameter_inputs("bulk_params")

            column_config = {
                "keep": st.column_config.CheckboxColumn("Import"),
                "subdevice": st.column_config.SelectboxColumn(
                    "Subdevice",
                    options=["single_bus", "single_ring_1", "single_ring_2", "vernier", "other"],
                ),
                "port": st.column_config.SelectboxColumn("Port", options=ALLOWED_PORTS),
            }

            if bulk_source == "Upload many CSV files":
                uploads = st.file_uploader(
                    "Select all raw/channel and trigger CSV files",
                    type=["csv"],
                    accept_multiple_files=True,
                    key="bulk_uploads",
                )
                upload_lookup = {upload.name: upload for upload in uploads}
                if uploads:
                    staging = paired_upload_rows(
                        list(upload_lookup),
                        default_device=default_device,
                        default_subdevice=default_subdevice,
                        default_port=default_port,
                        pair_triggers=pair_triggers,
                    )
                    edited = st.data_editor(
                        staging,
                        width="stretch",
                        num_rows="dynamic",
                        column_config={
                            **column_config,
                            "trigger_name": st.column_config.SelectboxColumn(
                                "Trigger file",
                                options=[""] + sorted(upload_lookup, key=sort_key),
                            ),
                        },
                        disabled=["raw_name"],
                        key="bulk_upload_editor",
                    )
                    if st.button("Import staged uploads", type="primary"):
                        inserted = add_records_from_upload_specs(
                            date=date,
                            batch_name=batch_name,
                            experiment_id=experiment_id,
                            rows=edited.to_dict("records"),
                            upload_lookup=upload_lookup,
                            shared_params=shared_params,
                        )
                        st.success(f"Inserted {len(inserted)} records into experiments/{slug(experiment_id)}/")
                else:
                    st.info("Select all related CSV files at once; edit device, subdevice, and port in the table before importing.")
            else:
                folder = st.text_input("Folder containing CSV files", value="")
                pattern = st.text_input("File pattern", value="*.csv")
                if folder:
                    source_dir = Path(folder)
                    if source_dir.exists():
                        paths = list(source_dir.glob(pattern))
                        if paths:
                            staging = paired_path_rows(
                                paths,
                                default_device=default_device,
                                default_subdevice=default_subdevice,
                                default_port=default_port,
                                pair_triggers=pair_triggers,
                            )
                            edited = st.data_editor(
                                staging,
                                width="stretch",
                                num_rows="dynamic",
                                column_config=column_config,
                                disabled=["raw_source_path"],
                                key="bulk_path_editor",
                            )
                            if st.button("Copy staged local files and import", type="primary"):
                                inserted = add_records_from_path_specs(
                                    date=date,
                                    batch_name=batch_name,
                                    experiment_id=experiment_id,
                                    rows=edited.to_dict("records"),
                                    shared_params=shared_params,
                                )
                                st.success(f"Inserted {len(inserted)} records into experiments/{slug(experiment_id)}/")
                        else:
                            st.warning("No CSV files matched that pattern.")
                    else:
                        st.warning("Folder does not exist.")
                else:
                    st.info("Enter a folder path containing oscilloscope CSV files.")

    with browse_tab:
        st.subheader("Browse / Edit Records")
        records = load_records(active_only=False)
        edited = st.data_editor(records, width="stretch", num_rows="dynamic")
        c1, c2 = st.columns(2)
        if c1.button("Save record table"):
            edited["updated_at"] = edited.get("updated_at", "")
            save_records(edited)
            st.success("Saved data/records.csv")
        delete_ids = c2.multiselect("Soft-delete record IDs", records["record_id"].tolist())
        if c2.button("Soft delete selected"):
            soft_delete_records(delete_ids)
            st.success("Marked selected records inactive.")


def spectrum_preview():
    st.header("Spectrum Preview")
    records = filtered_records(active_only=True)
    selected_ids = st.multiselect(
        "Select spectra",
        records["record_id"].tolist(),
        format_func=lambda rid: row_label(records[records["record_id"] == rid].iloc[0]),
    )
    if not selected_ids:
        st.info("Select one or more spectra.")
        return
    selected = records[records["record_id"].isin(selected_ids)]

    view_mode = st.radio("View", ["Raw power", "Single-bus ratio (dB)"], horizontal=True)
    baseline = None
    if view_mode.startswith("Single-bus ratio"):
        baseline = record_picker("Single-bus baseline", records, key="preview_baseline", only_port="single_bus")
        if baseline is None:
            return
    downsample = st.number_input("Downsample factor", min_value=1, max_value=500, value=10, key="preview_downsample")
    plot_mode = st.radio("Plot mode", ["Overlap", "Individual"], horizontal=True)

    coll = load_spectra(selected).downsample(int(downsample))
    if baseline is not None:
        base = load_spectrum(baseline).downsample(int(downsample))
        coll = coll.single_bus_ratio_by(base)

    fig = plot_overlap(coll, "Spectrum Preview") if plot_mode == "Overlap" else plot_individual(coll, "Spectrum Preview")
    st.pyplot(fig, clear_figure=True)

    if st.button("Export preview"):
        exp_id = selected.iloc[0]["experiment_id"]
        out_dir = experiment_dir(exp_id) / "preview" / f"preview_{now_iso().replace(':', '-')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_dir / "preview.png")
        for name, spec in coll.items():
            spec.to_dataframe().to_csv(out_dir / f"{slug(name)}.csv", index=False)
        st.success(f"Saved preview to {relative_to_project(out_dir)}")


def through_tmm_fitting():
    st.header("Through-Port TMM Fitting")
    records = filtered_records(active_only=True)
    through = records[records["port"] == "through"]
    if through.empty:
        st.warning("No active through-port records.")
        return

    target = record_picker("Through spectrum", through, key="fit_target")
    baseline = record_picker("Single-bus baseline", records, key="fit_baseline", only_port="single_bus")
    if target is None or baseline is None:
        return

    c1, c2, c3, c4 = st.columns(4)
    downsample = c1.number_input("Downsample", min_value=1, max_value=500, value=10)
    prominence = c2.number_input("Prominence dB", min_value=0.1, value=8.0, step=0.5)
    distance = c3.number_input("Distance pts", min_value=1, value=3000)
    fit_half_window = c4.number_input("Fit half-window pts", min_value=10, value=500)
    c5, c6, c7 = st.columns(3)
    radius_um = c5.number_input("Radius um", min_value=1.0, value=26.0, step=1.0)
    ng = c6.number_input("ng for Q", min_value=1.0, value=4.05, step=0.01)
    fsr_guess = c7.number_input("FSR fallback nm", min_value=0.01, value=2.5, step=0.1)

    spec = load_spectrum(target).downsample(int(downsample))
    base = load_spectrum(baseline).downsample(int(downsample))
    ratio_spectrum = SpectraCollection({"target": spec}).single_bus_ratio_by(base)["target"]
    candidates = detect_resonances(ratio_spectrum, prominence_db=float(prominence), distance_pts=int(distance), mode="dips")
    st.subheader("Peak Detection")
    st.pyplot(plot_detection(ratio_spectrum, candidates), clear_figure=True)
    edited_candidates = st.data_editor(candidates, width="stretch")
    st.caption(f"{int(edited_candidates['keep'].astype(bool).sum()) if not edited_candidates.empty else 0} kept / {len(edited_candidates)} detected")

    if st.button("Fit kept peaks", type="primary"):
        exp_id = target["experiment_id"]
        run_id = f"fit_{now_iso().replace(':', '-')}"
        out_dir = experiment_dir(exp_id) / "fits" / run_id
        params = FitParams(
            radius_um=float(radius_um),
            ng=float(ng),
            fsr_guess_nm=float(fsr_guess),
            fit_half_window_pts=int(fit_half_window),
        )
        audit = single_bus_ratio_audit(spec, base, ratio_spectrum)
        audit.update({
            "fit_model": "symmetric_double_bus_through_fixed_fsr",
            "radius_um": float(radius_um),
            "ng_for_q": float(ng),
            "fsr_fallback_nm": float(fsr_guess),
            "fit_half_window_pts": int(fit_half_window),
            "peak_prominence_db": float(prominence),
            "peak_distance_pts": int(distance),
            "downsample_factor": int(downsample),
        })
        results = fit_spectrum(ratio_spectrum, edited_candidates, params)
        save_fit_outputs(
            spectrum=ratio_spectrum,
            candidates=edited_candidates,
            results=results,
            output_dir=out_dir,
            baseline_record_id=baseline["record_id"],
            math_audit=audit,
        )
        st.success(f"Saved fit outputs to {relative_to_project(out_dir)}")
        st.dataframe(results, width="stretch")


def main():
    st.title("Microring Pipeline")
    mode = st.sidebar.radio("Mode", ["Database Management", "Analyze Data"])
    if mode == "Database Management":
        database_management()
    else:
        page = st.sidebar.radio("Analysis", ["Spectrum Preview", "Through-Port TMM Fitting"])
        if page == "Spectrum Preview":
            spectrum_preview()
        else:
            through_tmm_fitting()


if __name__ == "__main__":
    main()
