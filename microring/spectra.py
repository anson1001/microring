"""Spectrum loading and transformation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_LENGTH, END_NM, START_NM
from .data_store import project_path


@dataclass
class Spectrum:
    wavelength_nm: np.ndarray
    values: np.ndarray
    unit: str = "raw_power"
    metadata: dict = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        self.wavelength_nm = np.asarray(self.wavelength_nm, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        order = np.argsort(self.wavelength_nm)
        self.wavelength_nm = self.wavelength_nm[order]
        self.values = self.values[order]

    @property
    def transmission_dB(self) -> np.ndarray:
        if self.unit == "dB":
            return self.values
        return 10.0 * np.log10(np.clip(clean_power(self.values), 1e-15, None))

    @property
    def transmission_linear(self) -> np.ndarray:
        if self.unit == "dB":
            return 10.0 ** (self.values / 10.0)
        return clean_power(self.values)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "wavelength_nm": self.wavelength_nm,
            "transmission_dB" if self.unit == "dB" else "raw_power": self.values,
        })

    def downsample(self, factor: int) -> "Spectrum":
        factor = int(max(1, factor))
        if factor == 1:
            return self
        n = (len(self.values) // factor) * factor
        wl = self.wavelength_nm[:n].reshape(-1, factor).mean(axis=1)
        values = self.values[:n].reshape(-1, factor).mean(axis=1)
        return Spectrum(wl, values, unit=self.unit, metadata=dict(self.metadata), name=self.name)


class SpectraCollection:
    def __init__(self, spectra: dict[str, Spectrum] | None = None):
        self.spectra = spectra or {}

    def __getitem__(self, key: str) -> Spectrum:
        return self.spectra[key]

    def items(self):
        return self.spectra.items()

    def values(self):
        return self.spectra.values()

    @property
    def names(self) -> list[str]:
        return list(self.spectra.keys())

    def single_bus_ratio_by(self, baseline: Spectrum) -> "SpectraCollection":
        """Return device-to-single-bus power ratio in dB.

        The output is a measured ratio:

            ratio_dB = 10 * log10(P_device / P_single_bus)

        It is not forced to be <= 0 dB. Values above 0 dB mean the selected
        single-bus trace is lower than the device trace at that wavelength.
        """
        result = {}
        baseline_power = clean_power(baseline.transmission_linear)
        for name, spec in self.spectra.items():
            device_power = clean_power(spec.transmission_linear)
            if len(device_power) != len(baseline_power):
                raise ValueError("Spectrum and baseline must have the same length.")
            norm_db = 10.0 * np.log10(np.clip(device_power / (baseline_power + 1e-15), 1e-15, None))
            meta = dict(spec.metadata)
            meta["unit"] = "dB"
            meta["math_operation"] = "single_bus_ratio"
            meta["math_formula"] = "10*log10(P_device/P_single_bus)"
            meta["baseline_record_id"] = baseline.metadata.get("record_id", "")
            result[name] = Spectrum(spec.wavelength_nm.copy(), norm_db, unit="dB", metadata=meta, name=spec.name)
        return SpectraCollection(result)

    def normalize_by(self, baseline: Spectrum) -> "SpectraCollection":
        return self.single_bus_ratio_by(baseline)

    def self_normalize(self) -> "SpectraCollection":
        return SpectraCollection({
            name: Spectrum(spec.wavelength_nm.copy(), spec.values - np.nanmax(spec.values), "dB", dict(spec.metadata), spec.name)
            for name, spec in self.spectra.items()
        })

    def downsample(self, factor: int) -> "SpectraCollection":
        return SpectraCollection({name: spec.downsample(factor) for name, spec in self.spectra.items()})


def clean_power(power: np.ndarray) -> np.ndarray:
    p = np.asarray(power, dtype=float).copy()
    bad = ~np.isfinite(p) | (p <= 0)
    if not np.any(bad):
        return p
    good = ~bad
    if not np.any(good):
        raise ValueError("Power trace contains no positive finite samples.")
    x = np.arange(len(p))
    p[bad] = np.interp(x[bad], x[good], p[good])
    return p


def power_cleanup_stats(power: np.ndarray | list[float]) -> dict:
    p = np.asarray(power, dtype=float)
    nonfinite = ~np.isfinite(p)
    nonpositive = np.isfinite(p) & (p <= 0)
    bad = nonfinite | nonpositive
    return {
        "n_samples": int(len(p)),
        "n_nonfinite": int(np.sum(nonfinite)),
        "n_nonpositive": int(np.sum(nonpositive)),
        "n_interpolated": int(np.sum(bad)),
    }


def single_bus_ratio_audit(device: Spectrum, baseline: Spectrum, ratio: Spectrum) -> dict:
    device_raw_for_stats = device.values if device.unit != "dB" else device.transmission_linear
    baseline_raw_for_stats = baseline.values if baseline.unit != "dB" else baseline.transmission_linear
    device_power = clean_power(device_raw_for_stats)
    baseline_power = clean_power(baseline_raw_for_stats)
    ratio_linear = ratio.transmission_linear
    ratio_db = ratio.transmission_dB
    if len(device_power) != len(baseline_power):
        raise ValueError("Device and baseline traces must have the same length for audit.")

    device_stats = power_cleanup_stats(device_raw_for_stats)
    baseline_stats = power_cleanup_stats(baseline_raw_for_stats)
    quantiles = [0.0, 0.5, 1.0]
    ratio_linear_q = np.nanquantile(ratio_linear, quantiles)
    ratio_db_q = np.nanquantile(ratio_db, quantiles)
    return {
        "operation": "single_bus_ratio",
        "formula": "T_dB = 10*log10(P_device/P_single_bus)",
        "output_unit": "dB",
        "device_record_id": device.metadata.get("record_id", ""),
        "baseline_record_id": baseline.metadata.get("record_id", ""),
        "device_label": device.name,
        "baseline_label": baseline.name,
        "n_samples": int(len(ratio_linear)),
        "wavelength_start_nm": float(ratio.wavelength_nm[0]),
        "wavelength_stop_nm": float(ratio.wavelength_nm[-1]),
        "device_power_median": float(np.nanmedian(device_power)),
        "baseline_power_median": float(np.nanmedian(baseline_power)),
        "device_power_interpolated_samples": device_stats["n_interpolated"],
        "baseline_power_interpolated_samples": baseline_stats["n_interpolated"],
        "ratio_linear_min": float(ratio_linear_q[0]),
        "ratio_linear_median": float(ratio_linear_q[1]),
        "ratio_linear_max": float(ratio_linear_q[2]),
        "ratio_db_min": float(ratio_db_q[0]),
        "ratio_db_median": float(ratio_db_q[1]),
        "ratio_db_max": float(ratio_db_q[2]),
        "ratio_fraction_above_1": float(np.mean(ratio_linear > 1.0)),
        "note": "Ratio values above 1 are possible when the selected single-bus trace is lower than the device trace.",
    }


def wavelength_axis(n_points: int) -> np.ndarray:
    return np.linspace(START_NM, END_NM, n_points)


def read_numeric_trace(path: str | Path) -> np.ndarray:
    p = project_path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        df = pd.read_csv(p, skiprows=15)
        if df.shape[1] >= 2:
            return pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().to_numpy()
    except Exception:
        pass
    df = pd.read_csv(p)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError(f"No numeric columns found in {p}")
    return numeric.iloc[:, -1].dropna().to_numpy()


def read_triggered_trace(raw_path: str | Path, trigger_path: str | Path = "", data_length: int = DATA_LENGTH) -> np.ndarray:
    raw = read_numeric_trace(raw_path)
    if not trigger_path:
        return raw[:data_length]
    trigger = read_numeric_trace(trigger_path)
    crossings = np.where(trigger > 2.5)[0]
    if len(crossings) == 0:
        return raw[:data_length]
    start = int(crossings[-1])
    return raw[start:start + data_length]


def spectrum_name(record: pd.Series) -> str:
    label = str(record.get("label", "")).strip()
    if label:
        return label
    return f"{record.get('device')}_{record.get('subdevice')}_{record.get('port')}"


def load_spectrum(record: pd.Series, data_length: int = DATA_LENGTH) -> Spectrum:
    raw = read_triggered_trace(record["raw_file_path"], record.get("trigger_file_path", ""), data_length=data_length)
    return Spectrum(
        wavelength_axis(len(raw)),
        raw,
        unit="raw_power",
        metadata=record.to_dict(),
        name=spectrum_name(record),
    )


def load_spectra(records: pd.DataFrame, data_length: int = DATA_LENGTH) -> SpectraCollection:
    spectra = {}
    for _, row in records.iterrows():
        spec = load_spectrum(row, data_length=data_length)
        key = f"{row.get('record_id')}_{spec.name}"
        spectra[key] = spec
    return SpectraCollection(spectra)
