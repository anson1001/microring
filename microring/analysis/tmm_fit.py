"""Stepwise TMM fitting for normalized through-port spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..data_store import create_analysis_run
from ..spectra import Spectrum
from .metrics import (
    finesse,
    linewidth_nm_from_q,
    loaded_q_from_sum_rule,
    ng_from_fsr,
    propagation_loss_db_cm,
    q_from_amplitude_factor,
    total_external_q,
)
from .tmm_model import through_power_fixed_fsr


@dataclass
class FitParams:
    radius_um: float = 26.0
    ng: float = 4.05
    fsr_guess_nm: float = 2.5
    fit_half_window_pts: int = 500
    max_trials: int = 3


def candidate_fsr(candidates: pd.DataFrame, row_index: int, fsr_guess_nm: float) -> float:
    kept = candidates[candidates.get("keep", True).astype(bool)].sort_values("wavelength_nm").reset_index(drop=True)
    if len(kept) < 2:
        return fsr_guess_nm
    lambda0 = kept.loc[row_index, "wavelength_nm"]
    if row_index < len(kept) - 1:
        fsr = kept.loc[row_index + 1, "wavelength_nm"] - lambda0
    else:
        fsr = lambda0 - kept.loc[row_index - 1, "wavelength_nm"]
    return float(fsr) if np.isfinite(fsr) and fsr > 0 else fsr_guess_nm


def fit_resonance(
    spectrum: Spectrum,
    candidate: pd.Series,
    fsr_nm: float,
    params: FitParams,
) -> dict:
    idx0 = int(candidate["sample_index"])
    left = max(0, idx0 - params.fit_half_window_pts)
    right = min(len(spectrum.values), idx0 + params.fit_half_window_pts + 1)
    x = spectrum.wavelength_nm[left:right]
    y = 10.0 ** (spectrum.transmission_dB[left:right] / 10.0)
    lambda_guess = float(candidate["wavelength_nm"])
    b0_guess = float(np.nanmedian(y)) if len(y) else 1.0
    guesses = [(0.95, 0.95), (0.9, 0.98), (0.98, 0.9)][: max(1, params.max_trials)]
    bounds = ([0.01, 0.01, x.min(), 0.001, -10.0], [0.9999, 0.9999, x.max(), 10.0, 10.0])

    best = None
    best_rss = np.inf
    for t0, a0 in guesses:
        try:
            popt, pcov = curve_fit(
                lambda xx, t, a, lambda0, b0, b1: through_power_fixed_fsr(xx, t, a, lambda0, b0, b1, fsr_nm),
                x,
                y,
                p0=[t0, a0, lambda_guess, b0_guess, 0.0],
                bounds=bounds,
                maxfev=50000,
            )
            model = through_power_fixed_fsr(x, *popt, fsr_nm)
            rss = float(np.sum((y - model) ** 2))
            if rss < best_rss:
                best = (popt, pcov, rss)
                best_rss = rss
        except Exception:
            continue

    base = {
        "keep": True,
        "sample_index": idx0,
        "fit_left_index": int(left),
        "fit_right_index": int(right),
        "candidate_wavelength_nm": float(candidate["wavelength_nm"]),
        "FSR_nm": fsr_nm,
        "fit_status": "failed",
        "notes": "",
    }
    if best is None:
        return base

    popt, pcov, rss = best
    t, a, lambda0, b0, b1 = [float(v) for v in popt]
    q_i = q_from_amplitude_factor(a, lambda0, params.radius_um, params.ng)
    q_c_single = q_from_amplitude_factor(t, lambda0, params.radius_um, params.ng)
    q_e_total = total_external_q(q_c_single, q_c_single)
    q_l = q_from_amplitude_factor(a * t**2, lambda0, params.radius_um, params.ng)
    q_l_from_sum = loaded_q_from_sum_rule(q_i, q_e_total)
    kappa_sq = max(0.0, 1.0 - t**2)
    return {
        **base,
        "lambda0_nm": lambda0,
        "t": t,
        "a": a,
        "kappa_sq": kappa_sq,
        "Qi": q_i,
        "Qc_single": q_c_single,
        "Qe_total": q_e_total,
        "Ql": q_l,
        "Ql_from_sum_rule": q_l_from_sum,
        "linewidth_nm": linewidth_nm_from_q(lambda0, q_l),
        "loss_db_cm": propagation_loss_db_cm(a, params.radius_um),
        "finesse": finesse(lambda0, q_l, fsr_nm),
        "ng_from_fsr": ng_from_fsr(lambda0, fsr_nm, params.radius_um),
        "n_eff_est": np.nan,
        "RSS": rss,
        "B0": b0,
        "B1": b1,
        "fit_status": "ok",
    }


def fit_spectrum(spectrum: Spectrum, candidates: pd.DataFrame, params: FitParams) -> pd.DataFrame:
    kept = candidates[candidates["keep"].astype(bool)].sort_values("wavelength_nm").reset_index(drop=True)
    rows = []
    for i, cand in kept.iterrows():
        fsr_nm = candidate_fsr(kept, i, params.fsr_guess_nm)
        rows.append(fit_resonance(spectrum, cand, fsr_nm, params))
    return pd.DataFrame(rows)


def summarize_fit_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame([{"n_total": 0, "n_kept_ok": 0}])
    keep_mask = results["keep"].astype(bool) if "keep" in results else pd.Series(True, index=results.index)
    status_mask = results["fit_status"].eq("ok") if "fit_status" in results else pd.Series(False, index=results.index)
    kept = results[keep_mask & status_mask]
    numeric_cols = [
        "Ql",
        "Ql_from_sum_rule",
        "Qi",
        "Qc_single",
        "Qe_total",
        "a",
        "t",
        "loss_db_cm",
        "linewidth_nm",
        "FSR_nm",
        "finesse",
        "ng_from_fsr",
    ]
    summary = {"n_total": len(results), "n_kept_ok": len(kept)}
    for col in numeric_cols:
        if col in kept:
            values = pd.to_numeric(kept[col], errors="coerce")
            summary[f"{col}_mean"] = values.mean()
            summary[f"{col}_median"] = values.median()
            summary[f"{col}_std"] = values.std()
    return pd.DataFrame([summary])


def save_distribution_plots(results: pd.DataFrame, output_dir: Path) -> None:
    dist_dir = output_dir / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    if results.empty:
        return
    keep_mask = results["keep"].astype(bool) if "keep" in results else pd.Series(True, index=results.index)
    status_mask = results["fit_status"].eq("ok") if "fit_status" in results else pd.Series(False, index=results.index)
    kept = results[keep_mask & status_mask]
    for col in ["Ql", "Qi", "Qc_single", "Qe_total", "loss_db_cm", "linewidth_nm", "FSR_nm", "finesse"]:
        if col not in kept:
            continue
        values = pd.to_numeric(kept[col], errors="coerce").dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=20)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(dist_dir / f"{col}_hist.png", dpi=200)
        plt.close(fig)


def save_fit_outputs(
    *,
    spectrum: Spectrum,
    candidates: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: Path,
    baseline_record_id: str = "",
    math_audit: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_dir = output_dir / "local_fits"
    local_dir.mkdir(exist_ok=True)
    spectrum.to_dataframe().to_csv(output_dir / "single_bus_ratio_spectrum.csv", index=False)
    candidates.to_csv(output_dir / "peak_candidates.csv", index=False)
    results.to_csv(output_dir / "fit_results_raw.csv", index=False)
    if not results.empty and "keep" in results:
        kept = results[results["keep"].astype(bool)]
    else:
        kept = results
    kept.to_csv(output_dir / "fit_results_kept.csv", index=False)
    summary = summarize_fit_results(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    save_math_audit(output_dir, math_audit or {})
    save_overview_plots(spectrum, candidates, output_dir)
    save_local_fit_plots(spectrum, results, local_dir)
    save_distribution_plots(results, output_dir)
    create_analysis_run(
        record_id=str(spectrum.metadata.get("record_id", "")),
        baseline_record_id=baseline_record_id,
        run_type="through_tmm",
        output_folder=output_dir,
        status="complete",
    )
    return {"summary": summary, "output_dir": output_dir}


def math_audit_dataframe(audit: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"name": str(key), "value": "" if value is None else str(value)} for key, value in sorted(audit.items())]
    )


def math_audit_markdown(audit: dict) -> str:
    lines = [
        "# Math Audit",
        "",
        "## Spectrum Preprocessing",
        "",
        "- Raw oscilloscope samples are treated as power-like voltage.",
        "- Non-finite and non-positive power samples are linearly interpolated before logarithms.",
        "- Single-bus ratio is computed as `T_dB = 10*log10(P_device/P_single_bus)`.",
        "- Ratio values above 0 dB are allowed and mean `P_device > P_single_bus` for the selected traces.",
        "",
        "## Through-Port TMM Model",
        "",
        "`phi = 2*pi*(lambda - lambda0)/FSR`",
        "",
        "`T_ring = (t^2 + (a*t)^2 - 2*a*t^2*cos(phi)) / (1 + (a*t^2)^2 - 2*a*t^2*cos(phi))`",
        "",
        "`T_fit = (B0 + B1*(lambda - lambda0)) * T_ring`",
        "",
        "Here `t` is the self-coupling amplitude per coupler and `a` is the round-trip amplitude transmission.",
        "",
        "## Derived Metrics",
        "",
        "- `Qi` is computed from amplitude factor `a`.",
        "- `Qc_single` is computed from one coupler amplitude factor `t`.",
        "- `Qe_total` uses `1/Qe_total = 1/Qc1 + 1/Qc2`; for the symmetric model this is `Qc_single/2`.",
        "- `Ql` is computed from amplitude factor `a*t^2`.",
        "- `Ql_from_sum_rule` uses `1/Ql = 1/Qi + 1/Qe_total`.",
        "- `loss_db_cm = -20*log10(a)/(2*pi*radius_cm)`.",
        "- `ng_from_fsr = lambda0^2/(FSR*2*pi*radius)`; this is group index, not phase effective index.",
        "",
        "## Run Values",
        "",
    ]
    if audit:
        for key, value in sorted(audit.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No run-specific audit values were provided.")
    lines.append("")
    return "\n".join(lines)


def save_math_audit(output_dir: Path, audit: dict) -> None:
    math_audit_dataframe(audit).to_csv(output_dir / "math_audit.csv", index=False)
    (output_dir / "math_audit.md").write_text(math_audit_markdown(audit), encoding="utf-8")


def save_overview_plots(spectrum: Spectrum, candidates: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(spectrum.wavelength_nm, spectrum.values, lw=0.8)
    ax.set_title("Single-bus ratio spectrum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Single-bus ratio (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "preview_full_spectrum.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(spectrum.wavelength_nm, spectrum.values, lw=0.8, label=spectrum.name)
    if not candidates.empty:
        kept = candidates[candidates["keep"].astype(bool)]
        omitted = candidates[~candidates["keep"].astype(bool)]
        if not kept.empty:
            ax.scatter(kept["wavelength_nm"], kept["transmission_dB"], s=18, c="tab:red", label="kept candidates")
        if not omitted.empty:
            ax.scatter(omitted["wavelength_nm"], omitted["transmission_dB"], s=18, c="tab:gray", label="omitted candidates")
    ax.set_title("Peak detection")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Single-bus ratio (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "peak_detection.png", dpi=200)
    plt.close(fig)


def save_local_fit_plots(spectrum: Spectrum, results: pd.DataFrame, local_dir: Path) -> None:
    if results.empty:
        return
    ok = results[(results.get("fit_status", "") == "ok") & (results.get("keep", True).astype(bool))]
    for _, row in ok.iterrows():
        left = int(row.get("fit_left_index", max(0, int(row["sample_index"]) - 500)))
        right = int(row.get("fit_right_index", min(len(spectrum.values), int(row["sample_index"]) + 501)))
        x = spectrum.wavelength_nm[left:right]
        y_db = spectrum.transmission_dB[left:right]
        model_lin = through_power_fixed_fsr(
            x,
            float(row["t"]),
            float(row["a"]),
            float(row["lambda0_nm"]),
            float(row["B0"]),
            float(row["B1"]),
            float(row["FSR_nm"]),
        )
        model_db = 10.0 * np.log10(np.clip(model_lin, 1e-15, None))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, y_db, ".", ms=3, label="data")
        ax.plot(x, model_db, "-", lw=2, label="TMM fit")
        ax.set_title(f"{float(row['lambda0_nm']):.4f} nm, Ql={float(row['Ql']):.3g}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(local_dir / f"local_fit_{float(row['lambda0_nm']):.4f}nm.png", dpi=200)
        plt.close(fig)
