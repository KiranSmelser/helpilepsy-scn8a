import os
import pandas as pd

def save_per_patient(df, event_type, folder):
    """
    Save a DataFrame into per-patient CSVs using email as ID.
    """
    grouped = df.groupby('patient_email')

    for email, group in grouped:
        # Use a filesystem-safe version of the email as folder name
        safe_email = email.replace("@", "_at_").replace(".", "_")
        patient_folder = os.path.join(folder, safe_email)
        os.makedirs(patient_folder, exist_ok=True)

        filename = f"{event_type}.csv"
        filepath = os.path.join(patient_folder, filename)

        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            combined_df = pd.concat([existing_df, group], ignore_index=True).drop_duplicates()
        else:
            combined_df = group

        combined_df.to_csv(filepath, index=False)

        print(f"\n--- Saving intake data for: {safe_email} ---")
        print(f"Rows this batch: {len(group)}")
        print(f"Final file row count: {len(combined_df)}")
        print(f"File written: {filepath}")
