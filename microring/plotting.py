"""Matplotlib plotting helpers used by the Streamlit UI and exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .spectra import SpectraCollection, Spectrum


def plot_overlap(collection: SpectraCollection, title: str = "Spectra"):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, spec in collection.items():
        ax.plot(spec.wavelength_nm, spec.values, lw=0.8, label=name)
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (dB)" if all(s.unit == "dB" for s in collection.values()) else "Raw power")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    return fig


def plot_individual(collection: SpectraCollection, title: str = "Spectra"):
    n = max(1, len(collection.names))
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3, 2.8 * n)), squeeze=False)
    for ax, (name, spec) in zip(axes[:, 0], collection.items()):
        ax.plot(spec.wavelength_nm, spec.values, lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission (dB)" if spec.unit == "dB" else "Raw power")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_detection(spectrum: Spectrum, candidates: pd.DataFrame, title: str = "Detected resonances"):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(spectrum.wavelength_nm, spectrum.values, lw=0.8, label=spectrum.name)
    if not candidates.empty:
        kept = candidates[candidates["keep"].astype(bool)]
        omitted = candidates[~candidates["keep"].astype(bool)]
        if not kept.empty:
            ax.scatter(kept["wavelength_nm"], kept["transmission_dB"], s=18, c="tab:red", label="kept candidates", zorder=3)
        if not omitted.empty:
            ax.scatter(omitted["wavelength_nm"], omitted["transmission_dB"], s=18, c="tab:gray", label="omitted candidates", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
