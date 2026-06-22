"""Lightweight entrypoint for the redesigned microring package."""

from microring.data_store import ensure_data_layout, load_records


def main() -> None:
    ensure_data_layout()
    records = load_records(active_only=False)
    print("Microring pipeline is ready.")
    print(f"Records: {len(records)}")
    print("Run the UI with: streamlit run app.py")


if __name__ == "__main__":
    main()
