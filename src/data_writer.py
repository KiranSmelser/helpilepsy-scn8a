import os
from src import config, utils

def write_patient_data(patient_id, data_by_sheet):
    """Writes each data domain (seizure, medication, mood & sleep) to a CSV file in the patient’s folder."""
    patient_dir = utils.get_patient_dir(patient_id)
    utils.ensure_directory_exists(patient_dir)

    for sheet_name, df in data_by_sheet.items():
        folder_name = sheet_name.lower().replace(" ", "_").replace("&", "").strip()
        file_path = os.path.join(patient_dir, folder_name + ".csv")
        df.to_csv(file_path, index=False)