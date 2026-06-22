"""CSV-backed record and analysis-run store.

The project deliberately uses CSV as the source of truth.  This module keeps
all CSV reads/writes in one place so the UI and analysis code do not mutate
files ad hoc.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .config import (
    ANALYSIS_RUNS_CSV,
    DATA_DIR,
    EXPERIMENTS_DIR,
    LEGACY_RECORD_CSV,
    PROJECT_ROOT,
    RECORDS_CSV,
)

ALLOWED_PORTS = ["single_bus", "through", "drop_single", "drop_vernier", "other"]

CORE_RECORD_COLUMNS = [
    "record_id",
    "date",
    "batch_name",
    "device",
    "subdevice",
    "port",
    "label",
    "experiment_id",
    "raw_file_path",
    "trigger_file_path",
    "created_at",
    "updated_at",
    "active",
    "notes",
]

PARAM_COLUMNS = [
    "param_repeat",
    "param_voltage_v",
    "param_current_ma",
    "param_temperature_c",
    "param_to_voltage_v",
    "param_to_current_ma",
    "param_tec_voltage_v",
    "param_tec_current_ma",
]

ANALYSIS_RUN_COLUMNS = [
    "run_id",
    "record_id",
    "baseline_record_id",
    "run_type",
    "created_at",
    "output_folder",
    "status",
    "notes",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slug(value: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip())
    text = "_".join(part for part in text.split("_") if part)
    return text or "untitled"


def default_experiment_id(date: str, batch_name: str, device: str | None = None) -> str:
    """Return the folder-safe experiment ID used for copied data and outputs.

    The date is the default experiment folder because one fabrication batch can
    be measured across many days. Batch remains metadata for filtering.
    """
    clean_date = clean_value(date) or "unknown_date"
    clean_batch = clean_value(batch_name)
    if clean_date and clean_date != "unknown_date":
        return slug(clean_date)
    parts = [clean_batch, clean_value(device)]
    return slug("_".join(part for part in parts if part))


def project_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def relative_to_project(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def ensure_data_layout() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    if not RECORDS_CSV.exists():
        if LEGACY_RECORD_CSV.exists():
            migrate_legacy_records(LEGACY_RECORD_CSV, RECORDS_CSV)
        else:
            empty = pd.DataFrame(columns=CORE_RECORD_COLUMNS + PARAM_COLUMNS)
            empty.to_csv(RECORDS_CSV, index=False)
    if not ANALYSIS_RUNS_CSV.exists():
        pd.DataFrame(columns=ANALYSIS_RUN_COLUMNS).to_csv(ANALYSIS_RUNS_CSV, index=False)


def migrate_legacy_records(source: Path, destination: Path) -> None:
    legacy = pd.read_csv(source, on_bad_lines="warn")
    rows = []
    created = now_iso()
    for idx, row in legacy.iterrows():
        date = clean_value(row.get("date")) or "unknown_date"
        device = clean_value(row.get("device")) or "unknown_device"
        subdevice = clean_value(row.get("sub-device")) or "other"
        port = normalize_port(clean_value(row.get("port")) or "other")
        label = clean_value(row.get("name")) or f"{device}_{subdevice}_{port}"
        batch = (
            clean_value(row.get("batch_name"))
            or clean_value(row.get("batch"))
            or clean_value(row.get("experiment_id"))
            or date
        )
        experiment_id = default_experiment_id(date, batch)
        folder = clean_value(row.get("folder")) or ""
        raw_stem = clean_value(row.get("channel path")) or tek_stem(row.get("channel"))
        trigger_stem = clean_value(row.get("trigger path")) or tek_stem(row.get("trigger"))
        rows.append({
            "record_id": f"rec_{idx + 1:04d}",
            "date": date,
            "batch_name": batch,
            "device": device,
            "subdevice": subdevice,
            "port": port,
            "label": label,
            "experiment_id": experiment_id,
            "raw_file_path": legacy_file_path(folder, raw_stem),
            "trigger_file_path": legacy_file_path(folder, trigger_stem),
            "created_at": created,
            "updated_at": created,
            "active": True,
            "notes": clean_value(row.get("comments")),
            "param_repeat": clean_value(row.get("repeat")),
            "param_voltage_v": clean_value(row.get("EO (ring 1)")),
            "param_current_ma": None,
            "param_temperature_c": clean_value(row.get("temperature")),
            "param_to_voltage_v": clean_value(row.get("TO (V)")),
            "param_to_current_ma": clean_value(row.get("TO (I)")),
            "param_tec_voltage_v": clean_value(row.get("TEC (V)")),
            "param_tec_current_ma": clean_value(row.get("TEC (I)")),
        })
    pd.DataFrame(rows, columns=CORE_RECORD_COLUMNS + PARAM_COLUMNS).to_csv(destination, index=False)


def clean_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def tek_stem(number) -> str | None:
    if pd.isna(number):
        return None
    return f"TEK{int(float(number)):05d}"


def legacy_file_path(folder: str, stem: str | None) -> str:
    if not stem:
        return ""
    return (Path(folder) / f"{stem}.csv").as_posix()


def normalize_port(port: str) -> str:
    value = (port or "other").strip()
    if value in ALLOWED_PORTS:
        return value
    if value == "drop":
        return "drop_single"
    if value == "final_drop":
        return "drop_vernier"
    return "other"


def load_records(active_only: bool = False) -> pd.DataFrame:
    ensure_data_layout()
    df = pd.read_csv(RECORDS_CSV, dtype=str, keep_default_na=False)
    for col in CORE_RECORD_COLUMNS + PARAM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if active_only and "active" in df.columns:
        df = df[df["active"].astype(str).str.lower().isin(["true", "1", "yes"])]
    return df.reset_index(drop=True)


def save_records(records: pd.DataFrame) -> None:
    ensure_data_layout()
    df = records.copy()
    for col in CORE_RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    ordered = CORE_RECORD_COLUMNS + [c for c in df.columns if c not in CORE_RECORD_COLUMNS]
    df[ordered].to_csv(RECORDS_CSV, index=False)


def next_record_id(records: pd.DataFrame) -> str:
    return f"rec_{uuid.uuid4().hex[:10]}"


def experiment_dir(experiment_id: str) -> Path:
    root = EXPERIMENTS_DIR / slug(experiment_id)
    for child in ["raw", "preview", "fits", "exports"]:
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def copy_source_file(source: str | Path, dest_dir: Path, prefix: str, role: str) -> str:
    if not source:
        return ""
    src = project_path(source)
    if not src.exists():
        raise FileNotFoundError(f"File does not exist: {src}")
    dest = dest_dir / f"{prefix}_{role}{src.suffix or '.csv'}"
    shutil.copy2(src, dest)
    return relative_to_project(dest)


def save_upload_file(uploaded_file: BinaryIO, dest_dir: Path, prefix: str, role: str) -> str:
    if uploaded_file is None:
        return ""
    name = getattr(uploaded_file, "name", f"{role}.csv")
    suffix = Path(name).suffix or ".csv"
    dest = dest_dir / f"{prefix}_{role}{suffix}"
    with open(dest, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return relative_to_project(dest)


def build_record(
    *,
    date: str,
    batch_name: str,
    device: str,
    subdevice: str,
    port: str,
    label: str,
    raw_file_path: str,
    trigger_file_path: str = "",
    experiment_id: str | None = None,
    notes: str = "",
    params: dict | None = None,
) -> dict:
    created = now_iso()
    exp_id = experiment_id or default_experiment_id(date, batch_name, device)
    record_id = f"rec_{uuid.uuid4().hex[:10]}"
    record = {
        "record_id": record_id,
        "date": date,
        "batch_name": batch_name,
        "device": device,
        "subdevice": subdevice,
        "port": normalize_port(port),
        "label": label or f"{device}_{subdevice}_{port}",
        "experiment_id": exp_id,
        "raw_file_path": raw_file_path,
        "trigger_file_path": trigger_file_path,
        "created_at": created,
        "updated_at": created,
        "active": True,
        "notes": notes,
    }
    for col in PARAM_COLUMNS:
        record[col] = ""
    if params:
        for key, value in params.items():
            col = key if key.startswith("param_") else f"param_{key}"
            record[col] = value
    return record


def append_record(record: dict) -> dict:
    append_records([record])
    return record


def append_records(records_to_append: list[dict]) -> list[dict]:
    if not records_to_append:
        return []
    records = load_records(active_only=False)
    for record in records_to_append:
        for col in record:
            if col not in records.columns:
                records[col] = ""
    records = pd.concat([records, pd.DataFrame(records_to_append)], ignore_index=True)
    save_records(records)
    return records_to_append


def add_record_from_paths(
    *,
    date: str,
    batch_name: str,
    device: str,
    subdevice: str,
    port: str,
    label: str,
    raw_source_path: str | Path,
    trigger_source_path: str | Path | None = None,
    experiment_id: str | None = None,
    notes: str = "",
    params: dict | None = None,
) -> dict:
    exp_id = experiment_id or default_experiment_id(date, batch_name, device)
    rec_id = f"rec_{uuid.uuid4().hex[:10]}"
    raw_dir = experiment_dir(exp_id) / "raw"
    raw_path = copy_source_file(raw_source_path, raw_dir, rec_id, "raw")
    trigger_path = copy_source_file(trigger_source_path, raw_dir, rec_id, "trigger") if trigger_source_path else ""
    record = build_record(
        date=date,
        batch_name=batch_name,
        device=device,
        subdevice=subdevice,
        port=port,
        label=label,
        raw_file_path=raw_path,
        trigger_file_path=trigger_path,
        experiment_id=exp_id,
        notes=notes,
        params=params,
    )
    record["record_id"] = rec_id
    return append_record(record)


def add_record_from_uploads(
    *,
    date: str,
    batch_name: str,
    device: str,
    subdevice: str,
    port: str,
    label: str,
    raw_upload,
    trigger_upload=None,
    experiment_id: str | None = None,
    notes: str = "",
    params: dict | None = None,
) -> dict:
    exp_id = experiment_id or default_experiment_id(date, batch_name, device)
    rec_id = f"rec_{uuid.uuid4().hex[:10]}"
    raw_dir = experiment_dir(exp_id) / "raw"
    raw_path = save_upload_file(raw_upload, raw_dir, rec_id, "raw")
    trigger_path = save_upload_file(trigger_upload, raw_dir, rec_id, "trigger") if trigger_upload else ""
    record = build_record(
        date=date,
        batch_name=batch_name,
        device=device,
        subdevice=subdevice,
        port=port,
        label=label,
        raw_file_path=raw_path,
        trigger_file_path=trigger_path,
        experiment_id=exp_id,
        notes=notes,
        params=params,
    )
    record["record_id"] = rec_id
    return append_record(record)


def add_records_from_path_specs(
    *,
    date: str,
    batch_name: str,
    experiment_id: str | None,
    rows: list[dict],
    shared_params: dict | None = None,
) -> list[dict]:
    """Copy many local spectrum files and append all records in one CSV write."""
    exp_id = experiment_id or default_experiment_id(date, batch_name)
    raw_dir = experiment_dir(exp_id) / "raw"
    records = []
    for spec in rows:
        if not str(spec.get("keep", True)).lower() in {"true", "1", "yes"}:
            continue
        raw_source_path = clean_value(spec.get("raw_source_path"))
        if not raw_source_path:
            continue
        rec_id = f"rec_{uuid.uuid4().hex[:10]}"
        params = dict(shared_params or {})
        params.update({k: v for k, v in spec.items() if str(k).startswith("param_") and clean_value(v)})
        device = clean_value(spec.get("device")) or "unknown_device"
        subdevice = clean_value(spec.get("subdevice")) or "other"
        port = clean_value(spec.get("port")) or "other"
        raw_path = copy_source_file(raw_source_path, raw_dir, rec_id, "raw")
        trigger_source_path = clean_value(spec.get("trigger_source_path"))
        trigger_path = copy_source_file(trigger_source_path, raw_dir, rec_id, "trigger") if trigger_source_path else ""
        record = build_record(
            date=date,
            batch_name=batch_name,
            device=device,
            subdevice=subdevice,
            port=port,
            label=clean_value(spec.get("label")) or f"{device}_{subdevice}_{port}",
            raw_file_path=raw_path,
            trigger_file_path=trigger_path,
            experiment_id=exp_id,
            notes=clean_value(spec.get("notes")) or "",
            params=params,
        )
        record["record_id"] = rec_id
        records.append(record)
    return append_records(records)


def add_records_from_upload_specs(
    *,
    date: str,
    batch_name: str,
    experiment_id: str | None,
    rows: list[dict],
    upload_lookup: dict,
    shared_params: dict | None = None,
) -> list[dict]:
    """Save many uploaded spectra and append all records in one CSV write."""
    exp_id = experiment_id or default_experiment_id(date, batch_name)
    raw_dir = experiment_dir(exp_id) / "raw"
    records = []
    for spec in rows:
        if not str(spec.get("keep", True)).lower() in {"true", "1", "yes"}:
            continue
        raw_name = clean_value(spec.get("raw_name"))
        raw_upload = upload_lookup.get(raw_name)
        if raw_upload is None:
            continue
        rec_id = f"rec_{uuid.uuid4().hex[:10]}"
        params = dict(shared_params or {})
        params.update({k: v for k, v in spec.items() if str(k).startswith("param_") and clean_value(v)})
        device = clean_value(spec.get("device")) or "unknown_device"
        subdevice = clean_value(spec.get("subdevice")) or "other"
        port = clean_value(spec.get("port")) or "other"
        raw_path = save_upload_file(raw_upload, raw_dir, rec_id, "raw")
        trigger_name = clean_value(spec.get("trigger_name"))
        trigger_upload = upload_lookup.get(trigger_name) if trigger_name else None
        trigger_path = save_upload_file(trigger_upload, raw_dir, rec_id, "trigger") if trigger_upload else ""
        record = build_record(
            date=date,
            batch_name=batch_name,
            device=device,
            subdevice=subdevice,
            port=port,
            label=clean_value(spec.get("label")) or f"{device}_{subdevice}_{port}",
            raw_file_path=raw_path,
            trigger_file_path=trigger_path,
            experiment_id=exp_id,
            notes=clean_value(spec.get("notes")) or "",
            params=params,
        )
        record["record_id"] = rec_id
        records.append(record)
    return append_records(records)


def soft_delete_records(record_ids: list[str]) -> None:
    records = load_records(active_only=False)
    mask = records["record_id"].isin(record_ids)
    records.loc[mask, "active"] = False
    records.loc[mask, "updated_at"] = now_iso()
    save_records(records)


def load_analysis_runs() -> pd.DataFrame:
    ensure_data_layout()
    return pd.read_csv(ANALYSIS_RUNS_CSV, dtype=str, keep_default_na=False)


def create_analysis_run(
    *,
    record_id: str,
    baseline_record_id: str,
    run_type: str,
    output_folder: str | Path,
    status: str = "created",
    notes: str = "",
) -> dict:
    ensure_data_layout()
    run = {
        "run_id": f"run_{uuid.uuid4().hex[:10]}",
        "record_id": record_id,
        "baseline_record_id": baseline_record_id,
        "run_type": run_type,
        "created_at": now_iso(),
        "output_folder": relative_to_project(output_folder),
        "status": status,
        "notes": notes,
    }
    runs = load_analysis_runs()
    runs = pd.concat([runs, pd.DataFrame([run])], ignore_index=True)
    runs.to_csv(ANALYSIS_RUNS_CSV, index=False)
    return run
