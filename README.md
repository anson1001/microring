# Microring Pipeline

CSV-backed Streamlit app for microring experiment records, spectrum preview,
resonance detection, and through-port TMM fitting.

## Run

```powershell
cd <repo>
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

For trusted local-only use, enable direct local file/folder import before
starting Streamlit:

```powershell
$env:MICRORING_ALLOW_LOCAL_FILE_IMPORT = "1"
streamlit run app.py --server.port 8502
```

Leave this unset on a hosted/public server. Upload-based import remains
available either way.

The app has two modes:

- **Database Management**: insert spectra, copy raw files into the project,
  edit metadata, and soft-delete records.
- **Analyze Data**: preview raw spectra and single-bus ratios, detect through-port
  resonances, review candidate peaks, run TMM fits, and save outputs.

## Project Layout

```text
app.py                    Streamlit UI
main.py                   lightweight CLI/import entrypoint
microring/                new package code
data/records.csv          CSV source of truth, created on first run
data/analysis_runs.csv    analysis run registry
experiments/              copied raw files and analysis outputs
legacy/                   old research scripts kept for reference
```

## Data Convention

Raw oscilloscope values are treated as power-like voltages. Single-bus ratio
conversion uses:

```text
T_dB = 10 * log10(P_device / P_single_bus)
```

Non-positive raw samples are interpolated before logarithmic conversion. Ratio
values can exceed 0 dB when the selected single-bus trace is lower than the
device trace. Each TMM fitting run writes `math_audit.csv` and `math_audit.md`
with the exact formulas and run-specific preprocessing statistics.

## Inserting Data

Use **Database Management -> Insert Spectrum**.

- **Single File** is for one spectrum/channel CSV at a time.
- **Bulk Insert** is for many spectra collected under the same date and
  metadata. Enter the shared date and batch name first. By default the date
  becomes the experiment folder, for example:

```text
experiments/2026_06_18/
```

For trusted local oscilloscope folders, enable local import, choose **Use local folder**, keep the pattern as
`*.csv`, then edit the staging table. Files named like `TEK00022.csv` and
`TEK00023.csv` are auto-paired as raw/channel and trigger. Edit the device,
subdevice, port, label, and notes columns before pressing import.

## Row Convention

CSV display rows are one-based after the header. Pandas/code indices are
zero-based. The UI shows record IDs and labels to avoid this mismatch.
