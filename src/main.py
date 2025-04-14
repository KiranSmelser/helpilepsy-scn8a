from create_dataframes import (
    load_excel_data,
    clean_seizure_data,
    clean_medication_data,
    split_mood_sleep_data
)
from helpers import save_per_patient

def main():
    # Full file path (change to new file weekly)
    path = r"/Users/elysetanzer/Desktop/Python Library/Study_SCN8A_Export-21-03-2025.xlsx"
    # Load each sheet
    seizure_raw, medication_raw, mood_sleep_raw = load_excel_data(path)

    # Clean and organize
    seizure_df = clean_seizure_data(seizure_raw)
    medication_df = clean_medication_data(medication_raw)
    mood_df, sleep_df = split_mood_sleep_data(mood_sleep_raw)

    # Save per patient into /patients_data/
    save_per_patient(seizure_df, event_type="seizures")
    save_per_patient(mood_df, event_type="mood")
    save_per_patient(sleep_df, event_type="sleep")

    print("Patient data saved and updated!")

if __name__ == "__main__":
    main()
