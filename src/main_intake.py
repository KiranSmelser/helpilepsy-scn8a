import sys
import os
sys.path.append(os.path.dirname(__file__))

from create_intake_dataframes import load_intake_excel, clean_intake_confirmations_data
from intake_helper import save_per_patient

def main():
    # Update path to intake Excel file
    path = "/Users/elysetanzer/Desktop/Python Library/20250325_Medications Extract.xlsx"
    output_folder = "/Users/elysetanzer/Desktop/Python Library/patient_data"

    raw_intake_df = load_intake_excel(path)
    intake_df = clean_intake_confirmations_data(raw_intake_df)

    print("\nCleaned Intake Data Preview:")
    print(intake_df.head())
    print(f"Total cleaned rows: {len(intake_df)}")
    print(f"Unique patient emails: {intake_df['patient_email'].nunique()}")

    save_per_patient(intake_df, event_type="daily_intake", folder=output_folder)

    print("\nIntake pipeline complete.")

if __name__ == "__main__":
    main()
