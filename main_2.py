"""
main_2.py -- Clean pipeline for microring resonator spectral analysis.

All experiment metadata lives in record.csv (single source of truth).
Workflow:  Query -> Load -> Normalize -> (Quick Look | TMM Fit) -> Output

See end of file for usage examples.
"""

import os
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import * #read_data, moving_average, downsample_data
import TMM

# ============================================================
# 0. Constants
# ============================================================
SWEEP_RANGE_NM  = 100.0          # laser sweeps from 1260 -> 1360 nm
START_NM        = 1260.0
END_NM          = 1360.0
SWEEP_RATE      = 2.0            # nm / s
NUM_ROWS        = 2_000_000      # oscilloscope record length
TIME_AMOUNT     = 64             # number of time divisions
DATA_LENGTH     = int(NUM_ROWS / TIME_AMOUNT * SWEEP_RANGE_NM / SWEEP_RATE)

# Load the master record once at import time.
_record = pd.read_csv("record.csv", on_bad_lines="warn")


# ============================================================
# 1. Run -- output folder manager
# ============================================================
class Run:
    """
    Every analysis run gets its own folder.  Outputs (CSV, PNG, ...) are
    written inside `./<name>/`.

    Usage
    -----
    >>> run = Run("V8_EO_comparison")          # creates ./V8_EO_comparison/
    >>> run = Run.prompt()                     # asks for name interactively
    """

    def __init__(self, name: str):
        self.name = name
        self._dir = Path(name)
        if self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True, exist_ok=False)
        print(f"[Run] created folder: {self._dir.resolve()}/")

    @property
    def dir(self) -> Path:
        return self._dir

    def path(self, filename: str) -> Path:
        """Return `self.dir / filename`."""
        return self._dir / filename

    @staticmethod
    def prompt(prompt_text: str = "Enter a name for this run") -> "Run":
        """Create a Run interactively -- asks for a folder name."""
        name = input(f"{prompt_text}: ").strip()
        if not name:
            raise ValueError("Run name cannot be empty.")
        return Run(name)


# ============================================================
# 2. Spectrum -- a single spectrum with metadata
# ============================================================
@dataclass
class Spectrum:
    """
    Thin container for one optical transmission spectrum.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength axis (nm), monotonically increasing.
    transmission_dB : ndarray
        Transmission values in **dB**.  This is what TMM.fit() expects as
        its ``transmission_col``.
    metadata : dict
        The corresponding row(s) from record.csv and any extra tags.
    name : str
        Human-readable label used in plot legends and file names.
    """

    wavelength: np.ndarray
    transmission_dB: np.ndarray
    metadata: dict = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.transmission_dB = np.asarray(self.transmission_dB, dtype=float)
        # enforce ascending wavelength
        idx = np.argsort(self.wavelength)
        self.wavelength = self.wavelength[idx]
        self.transmission_dB = self.transmission_dB[idx]

    # -- derived properties ---------------------------------------------------
    @property
    def transmission_lin(self) -> np.ndarray:
        """Linear transmission = 10^(dB/10)."""
        return 10.0 ** (self.transmission_dB / 10.0)

    @property
    def start_nm(self) -> float:
        return float(self.wavelength[0])

    @property
    def end_nm(self) -> float:
        return float(self.wavelength[-1])

    def __len__(self) -> int:
        return len(self.wavelength)

    # -- helpers --------------------------------------------------------------
    def copy(self) -> "Spectrum":
        """Deep copy (data + metadata)."""
        return Spectrum(
            self.wavelength.copy(),
            self.transmission_dB.copy(),
            dict(self.metadata),
            self.name,
        )

    def clipped(self, wl_min: float, wl_max: float) -> "Spectrum":
        """Return a new Spectrum trimmed to [wl_min, wl_max]."""
        mask = (self.wavelength >= wl_min) & (self.wavelength <= wl_max)
        return Spectrum(
            self.wavelength[mask],
            self.transmission_dB[mask],
            dict(self.metadata),
            self.name,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Two-column DataFrame: wavelength_nm, transmission_dB."""
        return pd.DataFrame({
            "wavelength_nm":   self.wavelength,
            "transmission_dB": self.transmission_dB,
        })


# ============================================================
# 3. SpectraCollection -- a group of spectra with a common x-axis
# ============================================================
class SpectraCollection:
    """
    Holds multiple named Spectrum objects that share the same wavelength axis
    (guaranteed by the load / normalize pipeline).

    Most operations (normalize, downsample, plot, to_csv) act on every
    spectrum in the collection at once.
    """

    def __init__(self, spectra: Optional[dict[str, Spectrum]] = None):
        self._spectra: dict[str, Spectrum] = spectra or {}

    # -- dict-like access -----------------------------------------------------
    def __getitem__(self, name: str) -> Spectrum:
        return self._spectra[name]

    def __setitem__(self, name: str, spec: Spectrum):
        self._spectra[name] = spec

    def __contains__(self, name: str) -> bool:
        return name in self._spectra

    def __len__(self) -> int:
        return len(self._spectra)

    def __iter__(self):
        return iter(self._spectra)

    def items(self):
        return self._spectra.items()

    def keys(self):
        return self._spectra.keys()

    def values(self):
        return self._spectra.values()

    @property
    def names(self) -> list[str]:
        return list(self._spectra.keys())

    @property
    def wavelength(self) -> np.ndarray:
        """Shared wavelength axis (from first spectrum)."""
        if not self._spectra:
            raise ValueError("SpectraCollection is empty.")
        return next(iter(self._spectra.values())).wavelength

    # -- transformations (return new collection) ------------------------------
    def normalize_by(self, baseline: Spectrum) -> "SpectraCollection":
        """
        Normalise every spectrum by a single-bus (or other) baseline.

        Computes  T_norm(dB) = 10 * log10( T_lin_device / T_lin_baseline ).

        The baseline spectrum must share the same wavelength axis.
        """
        if len(baseline.wavelength) != len(self.wavelength):
            raise ValueError(
                f"Baseline length {len(baseline.wavelength)} != "
                f"collection length {len(self.wavelength)}."
            )
        result = {}
        baseline_lin = baseline.transmission_lin
        for name, spec in self._spectra.items():
            ratio = spec.transmission_lin / (baseline_lin + 1e-15)
            result[name] = Spectrum(
                spec.wavelength.copy(),
                10.0 * np.log10(np.clip(ratio, 1e-15, None)),
                dict(spec.metadata),
                spec.name or name,
            )
        return SpectraCollection(result)

    def self_normalize(self) -> "SpectraCollection":
        """
        Normalise each spectrum to its own maximum (peak = 0 dB).
        Good for comparing resonance shapes independent of insertion loss.
        """
        result = {}
        for name, spec in self._spectra.items():
            db = spec.transmission_dB - np.max(spec.transmission_dB)
            result[name] = Spectrum(
                spec.wavelength.copy(),
                db,
                dict(spec.metadata),
                spec.name or name,
            )
        return SpectraCollection(result)

    def downsample(self, factor: int = 10) -> "SpectraCollection":
        """
        Boxcar-average downsampling.  Returns a new collection.
        """
        # build a temporary 2-D array for downsample_data()
        names = self.names
        arr = np.array([self._spectra[n].transmission_dB for n in names])
        arr_ds = downsample_data(arr, points=factor)
        wl_ds = downsample_data(
            self.wavelength.reshape(1, -1), points=factor
        )[0]

        result = {}
        for i, name in enumerate(names):
            result[name] = Spectrum(
                wl_ds.copy(),
                arr_ds[i],
                dict(self._spectra[name].metadata),
                self._spectra[name].name or name,
            )
        return SpectraCollection(result)

    def clipped(self, wl_min: float, wl_max: float) -> "SpectraCollection":
        """Trim all spectra to a wavelength window."""
        result = {}
        for name, spec in self._spectra.items():
            result[name] = spec.clipped(wl_min, wl_max)
        return SpectraCollection(result)

    # -- I/O ------------------------------------------------------------------
    def to_comparison_csv(self, path: Union[str, Path]):
        """
        Save a multi-column CSV.  First column = wavelength_nm, subsequent
        columns = one per spectrum (column header = spectrum name).
        """
        cols = [self.wavelength]
        headers = ["wavelength_nm"]
        for name in self.names:
            cols.append(self._spectra[name].transmission_dB)
            headers.append(name)
        np.savetxt(
            str(path),
            np.column_stack(cols),
            delimiter=",",
            header=",".join(headers),
            comments="",
        )
        print(f"Saved comparison CSV: {path}")

    # -- plotting -------------------------------------------------------------
    def plot(
        self,
        title: str = "Transmission Spectra",
        run: Optional[Run] = None,
        filename: str = "comparison_plot.png",
        show: bool = True,
        dpi: int = 200,
        figsize: tuple = (12, 5),
    ):
        """
        Overlay all spectra in dB on one figure.

        If *run* is given, the figure is saved to ``run.path(filename)``.
        Set *show* = False to suppress the interactive window.
        """
        fig, ax = plt.subplots(figsize=figsize)
        for name, spec in self._spectra.items():
            ax.plot(spec.wavelength, spec.transmission_dB, lw=0.6, label=name)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission (dB)")
        ax.set_title(title)
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if run is not None:
            path = run.path(filename)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved plot: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_subplots(
        self,
        title: str = "Individual Spectra",
        run: Optional[Run] = None,
        filename: str = "subplots.png",
        show: bool = True,
        dpi: int = 200,
        ncols: int = 2,
    ):
        """
        One subplot per spectrum.  Useful when spectra have very different
        amplitudes.
        """
        n = len(self._spectra)
        nrows = max(1, int(np.ceil(n / ncols)))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 3 * nrows),
            squeeze=False,
        )
        for ax, (name, spec) in zip(axes.flat, self._spectra.items()):
            ax.plot(spec.wavelength, spec.transmission_dB, lw=0.6)
            ax.set_title(name, fontsize="small")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Transmission (dB)")
            ax.grid(True, alpha=0.3)
        # hide unused subplots
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        fig.suptitle(title)
        fig.tight_layout()

        if run is not None:
            path = run.path(filename)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved plot: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# 4. Query -- find data in record.csv
# ============================================================
def query(
    device: Optional[str] = None,
    sub_device: Optional[str] = None,
    port: Optional[str] = None,
    eo_ring1: Optional[float] = None,
    eo_ring2: Optional[float] = None,
    temperature: Optional[float] = None,
    date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return rows from record.csv matching ALL supplied (non-None) criteria.

    Examples
    --------
    >>> query(device="V8", sub_device="single_ring_1", port="through")
    >>> query(device="V1", eo_ring1=0)
    >>> query(date="23_4_2026")   # all TO / TEC measurements from that day
    """
    mask = pd.Series(True, index=_record.index)
    if device is not None:
        mask &= _record["device"] == device
    if sub_device is not None:
        mask &= _record["sub-device"] == sub_device
    if port is not None:
        mask &= _record["port"] == port
    if eo_ring1 is not None:
        mask &= _record["EO (ring 1)"] == eo_ring1
    if eo_ring2 is not None:
        mask &= _record["EO (ring 2)"] == eo_ring2
    if temperature is not None:
        mask &= _record["temperature"] == temperature
    if date is not None:
        mask &= _record["date"] == date
    return _record[mask].reset_index(drop=True)


def get_row(index: int) -> pd.Series:
    """Return a single row of record.csv by its integer index."""
    return _record.iloc[index]


def show_devices():
    """Print a summary of available devices in the record."""
    print("Devices in record.csv:")
    for dev in sorted(_record["device"].dropna().unique()):
        subdevs = _record[_record["device"] == dev]["sub-device"].dropna().unique()
        ports   = _record[_record["device"] == dev]["port"].dropna().unique()
        print(f"  {dev:12s}  sub: {list(subdevs)}  port: {list(ports)}")


# ============================================================
# 5. Loading -- read raw TEK files -> Spectrum
# ============================================================

def _wavelength_axis(n_points: int) -> np.ndarray:
    """Linearly spaced wavelength axis from START_NM to END_NM."""
    return np.linspace(START_NM, END_NM, n_points)


def load_spectrum(row: pd.Series, data_length: int = DATA_LENGTH) -> Spectrum:
    """
    Read one raw TEK trace from disk and return a Spectrum.

    The returned Spectrum contains **raw photodetector voltage** as its
    ``transmission_dB`` field (misnamed until normalisation -- the field
    stores whatever the current y-values are).  The Spectrum is tagged
    ``unit="raw"`` in metadata so downstream code can check.

    Parameters
    ----------
    row : pd.Series
        One row from record.csv (or a query result).
    data_length : int
        Number of samples to keep after the trigger.

    Returns
    -------
    Spectrum   (wavelength axis + raw voltage)
    """
    channel_path = str(row["folder"]) + str(row["channel path"]) + ".csv"
    trigger_path = str(row["folder"]) + str(row["trigger path"]) + ".csv"

    raw = read_data(channel_path, trigger_path, data_length=data_length)
    wl  = _wavelength_axis(len(raw))

    # Build a metadata dict from the record row
    meta = row.to_dict()
    name = _make_name(row)

    return Spectrum(wavelength=wl, transmission_dB=raw, metadata=meta, name=name)


def load_collection(
    rows: pd.DataFrame,
    data_length: int = DATA_LENGTH,
) -> SpectraCollection:
    """
    Load every row in *rows* and return a SpectraCollection.

    Each spectrum is named by its row index by default; call
    :meth:`SpectraCollection.normalize_by` with a single-bus Spectrum to
    convert raw voltage -> transmission in dB.
    """
    coll = SpectraCollection()
    for _, row in rows.iterrows():
        spec = load_spectrum(row, data_length=data_length)
        coll[spec.name] = spec
    return coll


def _make_name(row: pd.Series) -> str:
    """
    Auto-generate a short label for a spectrum from its record row.

    Priority:  ``name`` column  >  ``short form`` column  >
    ``{device}_{sub_device}_{port}_{EO/temp}``
    """
    if pd.notna(row.get("name")) and str(row["name"]).strip():
        return str(row["name"]).strip()
    if pd.notna(row.get("short form")) and str(row["short form"]).strip():
        return str(row["short form"]).strip()

    parts = [str(row["device"])]
    sub = row.get("sub-device")
    if pd.notna(sub):
        parts.append(str(sub))
    port = row.get("port")
    if pd.notna(port):
        parts.append(str(port))

    eo = row.get("EO (ring 1)")
    if pd.notna(eo) and float(eo) != 0:
        parts.append(f"EO{float(eo):.1f}V")
    temp = row.get("temperature")
    if pd.notna(temp) and float(temp) != 0:
        parts.append(f"T{float(temp):.1f}C")

    return "_".join(parts)


# ============================================================
# 6. Pipeline -- the high-level workflows
# ============================================================

def quick_look(
    rows: pd.DataFrame,
    single_bus_row: Union[int, pd.Series, None] = None,
    run: Optional[Run] = None,
    downsample: int = 1,
    show: bool = True,
    title: str = "Quick Look",
    self_norm: bool = False,
) -> SpectraCollection:
    """
    Load & quickly overlay spectra on one plot.

    This is the "preliminary check" -- no TMM fitting, just see the data.

    Parameters
    ----------
    rows : DataFrame
        Rows from record.csv (via :func:`query`).
    single_bus_row : int, Series, or None
        Row index (or row itself) of the single-bus reference.
        If None, no normalisation is applied.
    run : Run or None
        If given, the overlay plot is saved to the run folder.
    downsample : int
        Downsampling factor (1 = no downsampling).
    show : bool
        Whether to display the plot interactively.
    title : str
        Plot title.
    self_norm : bool
        If True, normalise each spectrum to its own max (0 dB).

    Returns
    -------
    SpectraCollection
        The loaded (and possibly normalised) spectra.
    """
    coll = load_collection(rows)

    # Normalize by single bus
    if single_bus_row is not None:
        if isinstance(single_bus_row, (int, np.integer)):
            sb_spec = load_spectrum(_record.iloc[single_bus_row])
        else:
            sb_spec = load_spectrum(single_bus_row)
        coll = coll.normalize_by(sb_spec)

    # Optional self-normalization
    if self_norm:
        coll = coll.self_normalize()

    # Downsample
    if downsample > 1:
        coll = coll.downsample(factor=downsample)

    # Plot
    coll.plot(title=title, run=run, show=show)

    # Save comparison CSV if run provided
    if run is not None:
        coll.to_comparison_csv(run.path("comparison_data.csv"))

    return coll


def tmm_fit(
    spectrum: Spectrum,
    run: Optional[Run] = None,
    ring_radius_um: float = 26.0,
    ng: float = 4.05,
    show_plots: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Run TMM fitting on a **single** spectrum.

    This writes a temporary CSV, calls ``TMM.fit()``, then (optionally)
    copies the TMM output folder into *run*'s folder.

    Parameters
    ----------
    spectrum : Spectrum
        A single spectrum whose ``transmission_dB`` is ready for TMM.
    run : Run or None
        Output folder manager.
    ring_radius_um : float
        Microring radius in µm.
    ng : float
        Group index for Q calculation.
    show_plots : bool
        Passed to TMM.fit -- whether to pop up each figure.
    **kwargs
        Forwarded to TMM.fit (e.g. fsr_guess_nm, min_prominence_db, ...).

    Returns
    -------
    pd.DataFrame
        The ``results_df`` from TMM.fit (one row per fitted resonance).
    """
    if run is None:
        run = Run.prompt("Enter run name for this TMM fit")

    # Write temporary CSV in the run folder
    tmp_csv = run.path("_input.csv")
    spec_df = spectrum.to_dataframe()
    spec_df.to_csv(tmp_csv, index=False)

    # Run TMM
    print(f"\n{'='*60}")
    print(f"TMM fitting: {spectrum.name}")
    print(f"{'='*60}")
    results_df = TMM.fit(
        file_path=str(tmp_csv),
        ring_radius_um=ring_radius_um,
        ng_for_Q=ng,
        show_plots=show_plots,
        **kwargs,
    )

    # Move TMM's own output folder into our run folder
    tmm_dir = Path(tmp_csv.stem)          # TMM.fit creates this
    if tmm_dir.exists():
        dest = run.path(tmm_dir.name)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(tmm_dir), str(dest))
        print(f"Moved TMM outputs -> {dest}/")

    # Save our own copy of the results
    results_csv = run.path("tmm_fit_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved results DataFrame -> {results_csv}")

    # Cleanup temp file
    os.remove(tmp_csv)

    return results_df


def batch_tmm(
    rows: pd.DataFrame,
    single_bus_row: Union[int, pd.Series],
    run: Optional[Run] = None,
    ring_radius_um: float = 26.0,
    ng: float = 4.05,
    downsample: int = 10,
    show_plots: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Load several spectra, normalise by a single bus, and run TMM on each.

    Collects per-spectrum **average** fit parameters into a summary table.

    Parameters
    ----------
    rows : DataFrame
        Rows from record.csv -- one spectrum per row.
    single_bus_row : int or Series
        Single-bus reference row (index into record.csv, or the row itself).
    run : Run or None
        Output folder; prompted if not given.
    ring_radius_um : float
    ng : float
    downsample : int
        Downsampling factor before TMM fitting.
    show_plots : bool
    **kwargs
        Passed to TMM.fit (fsr_guess_nm, min_prominence_db, etc.).

    Returns
    -------
    pd.DataFrame
        Summary table with columns: name, a, t, Qi, Qc_single, Ql, N_resonances.
    """
    if run is None:
        run = Run.prompt("Enter run name for batch TMM")

    # Load single-bus reference
    if isinstance(single_bus_row, (int, np.integer)):
        sb_row = _record.iloc[single_bus_row]
    else:
        sb_row = single_bus_row
    sb_spec = load_spectrum(sb_row)

    # Load & normalize all device spectra
    coll = load_collection(rows)
    coll = coll.normalize_by(sb_spec)

    if downsample > 1:
        coll = coll.downsample(factor=downsample)

    # Comparison plot of all input spectra
    coll.plot(title="Batch TMM -- Input Spectra", run=run, show=show_plots)

    # Fit each spectrum
    summary_rows = []
    for name, spec in coll.items():
        print(f"\n--- TMM for: {name} ---")
        try:
            res = tmm_fit(
                spec,
                run=Run(run.path(f"TMM_{name}").as_posix()),
                ring_radius_um=ring_radius_um,
                ng=ng,
                show_plots=show_plots,
                **kwargs,
            )
            # Average parameters (safe)
            avg = _safe_averages(res)
            avg["name"] = name
            avg["N_resonances"] = len(res)
            summary_rows.append(avg)
        except Exception as e:
            print(f"  !! TMM failed for {name}: {e}")
            summary_rows.append({
                "name": name, "a": np.nan, "t": np.nan,
                "Qi": np.nan, "Qc_single": np.nan, "Ql": np.nan,
                "N_resonances": 0,
            })

    summary = pd.DataFrame(summary_rows)
    summary_path = run.path("batch_tmm_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nBatch summary saved -> {summary_path}")
    print(summary.to_string(index=False))

    return summary


# ============================================================
# 7. Helpers
# ============================================================
def _safe_mean(series: pd.Series, positive_only: bool = False) -> float:
    """Robust mean -- drops NaN, Inf, and optionally values <= 0."""
    s = pd.to_numeric(series, errors="coerce")
    if positive_only:
        s = s[s > 0]
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.nan
    return float(np.mean(s))


def _safe_averages(results_df: pd.DataFrame) -> dict:
    """Return a dict of average a, t, Qi, Qc_single, Ql from TMM results."""
    return {
        "a":          _safe_mean(results_df["a"]),
        "t":          _safe_mean(results_df["t"]),
        "Qi":         _safe_mean(results_df["Qi"], positive_only=True),
        "Qc_single":  _safe_mean(results_df["Qc_single"], positive_only=True),
        "Ql":         _safe_mean(results_df["Ql"], positive_only=True),
    }


# ============================================================
# 8. Usage examples  (run this file directly)
# ============================================================
if __name__ == "__main__":
    # Show what's available
    # show_devices()

    # --- Example 1: Quick look at V8 ring 1 through port, various EO ---
    # rows = query(device="V8", sub_device="single_ring_1", port="through")
    # coll = quick_look(rows, single_bus_row=0, run=Run("V8_quick_look"))

    # --- Example 2: TMM fit a single spectrum ---
    # row = get_row(10)   # V8 single ring 1 through, 0V
    # spec = load_spectrum(row)
    # sb   = load_spectrum(get_row(0))   # single bus
    # spec_norm = SpectraCollection({"test": spec}).normalize_by(sb)["test"]
    # res = tmm_fit(spec_norm, run=Run("V8_single_TMM"), ring_radius_um=26)

    # --- Example 3: Batch TMM all EO voltages for V8 ---
    # rows = query(device="V8", sub_device="single_ring_1", port="through")
    # summary = batch_tmm(rows, single_bus_row=0, run=Run("V8_EO_TMM_batch"))

    # --- Example 4: TO / temperature sweep ---
    # rows = query(date="23_4_2026")
    # summary = batch_tmm(rows, single_bus_row=50, run=Run("TO_sweep_TMM"))

    # print("\nmain_2.py loaded. Use query(), load_spectrum(), quick_look(), tmm_fit(), batch_tmm().")

    # rows = query(date='16_6_2026', device="V10", sub_device="single_ring_2", port="through")
    # coll = quick_look(rows, single_bus_row=65, run=Run("V10_quick_look"))

    row = get_row(68)   # V10 single_ring_2 drop
    spec = load_spectrum(row)
    sb   = load_spectrum(get_row(65))   # single bus (row 64, NOT 65)
    spec_norm = SpectraCollection({"test": spec}).normalize_by(sb)["test"]
    res = tmm_fit(spec_norm, run=Run("V10_ring2_drop_TMM"), ring_radius_um=27)


