import os
import pandas as pd
from src import config, utils

def update_patient_data(patient_id, new_data):
    """Appends new data to the patient’s existing data or creates a new file."""
    updated_data = {}

    for sheet_name, df_new in new_data.items():
        folder_name = sheet_name.lower().replace(" ", "_").replace("&", "").strip()
        file_path = os.path.join(utils.get_patient_dir(patient_id), folder_name + ".csv")

        if os.path.exists(file_path):
            df_old = pd.read_csv(file_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates()
        else:
            df_combined = df_new

        updated_data[sheet_name] = df_combined

    return updated_data