"""Derived resonator metrics."""

from __future__ import annotations

import numpy as np


def q_from_amplitude_factor(amplitude_factor: float, lambda0_nm: float, radius_um: float, ng: float) -> float:
    factor = float(amplitude_factor)
    if not np.isfinite(factor) or factor <= 0 or factor >= 1:
        return np.nan
    length_um = 2.0 * np.pi * radius_um
    lambda_um = lambda0_nm / 1000.0
    return float(-2.0 * np.pi * ng * length_um / (lambda_um * np.log(factor**2)))


def total_external_q(*coupler_qs: float) -> float:
    values = np.asarray(coupler_qs, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return np.nan
    inv_total = np.sum(1.0 / values)
    return float(1.0 / inv_total) if inv_total > 0 else np.nan


def loaded_q_from_sum_rule(q_intrinsic: float, q_external_total: float) -> float:
    qi = float(q_intrinsic)
    qe = float(q_external_total)
    if not np.isfinite(qi) or not np.isfinite(qe) or qi <= 0 or qe <= 0:
        return np.nan
    return float(1.0 / (1.0 / qi + 1.0 / qe))


def linewidth_nm_from_q(lambda0_nm: float, q_loaded: float) -> float:
    if not np.isfinite(lambda0_nm) or not np.isfinite(q_loaded) or lambda0_nm <= 0 or q_loaded <= 0:
        return np.nan
    return float(lambda0_nm / q_loaded)


def propagation_loss_db_cm(a: float, radius_um: float) -> float:
    if not np.isfinite(a) or a <= 0:
        return np.nan
    length_cm = 2.0 * np.pi * radius_um * 1e-4
    return float(-20.0 * np.log10(a) / length_cm)


def ng_from_fsr(lambda0_nm: float, fsr_nm: float, radius_um: float) -> float:
    if not np.isfinite(fsr_nm) or fsr_nm <= 0:
        return np.nan
    length_nm = 2.0 * np.pi * radius_um * 1000.0
    return float(lambda0_nm**2 / (fsr_nm * length_nm))


def finesse(lambda0_nm: float, q_loaded: float, fsr_nm: float) -> float:
    if not np.isfinite(q_loaded) or not np.isfinite(fsr_nm) or lambda0_nm <= 0:
        return np.nan
    linewidth_nm = lambda0_nm / q_loaded
    return float(fsr_nm / linewidth_nm) if linewidth_nm > 0 else np.nan
