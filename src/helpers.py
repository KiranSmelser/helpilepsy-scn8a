import os
import pandas as pd

def save_per_patient(df, event_type, folder='patient_data'):
    for patient_email, group in df.groupby('patient_email'):
        patient_name = group['patient_name'].iloc[0].strip().replace(" ", "_") or "unknown"
        patient_folder = os.path.join(folder, patient_name)
        os.makedirs(patient_folder, exist_ok=True)

        filename = os.path.join(patient_folder, f"{event_type}.csv")

        # If file exists, read and append without duplicates
        if os.path.exists(filename):
            existing = pd.read_csv(filename, parse_dates=True)
            combined = pd.concat([existing, group], ignore_index=True)
            combined = combined.drop_duplicates()
        else:
            combined = group

        combined.to_csv(filename, index=False)
