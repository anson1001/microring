"""Resonance candidate detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..spectra import Spectrum


def detect_resonances(
    spectrum: Spectrum,
    prominence_db: float = 4.0,
    distance_pts: int = 2000,
    mode: str = "dips",
) -> pd.DataFrame:
    mode = mode.lower()
    if mode not in {"dips", "peaks", "both"}:
        raise ValueError("mode must be 'dips', 'peaks', or 'both'.")
    y = spectrum.transmission_dB
    rows = []

    def collect(kind: str, signal: np.ndarray) -> None:
        indices, props = find_peaks(signal, prominence=prominence_db, distance=distance_pts)
        for idx, prom in zip(indices, props.get("prominences", np.full(len(indices), np.nan))):
            rows.append({
                "keep": True,
                "kind": kind,
                "sample_index": int(idx),
                "wavelength_nm": float(spectrum.wavelength_nm[idx]),
                "transmission_dB": float(y[idx]),
                "prominence_db": float(prom),
                "notes": "",
            })

    if mode in {"dips", "both"}:
        collect("dip", -y)
    if mode in {"peaks", "both"}:
        collect("peak", y)
    return pd.DataFrame(rows).sort_values("wavelength_nm").reset_index(drop=True)
