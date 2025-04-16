import os
from src import config, data_loader, data_updater, data_writer

def main():
    raw_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith(".xlsx")] # assumes, for now, that there is just one file in raw/
    if not raw_files:
        print("No new raw data found.")
        return

    file_path = os.path.join(config.RAW_DATA_DIR, raw_files[0])
    patient_data = data_loader.load_weekly_data(file_path)

    for patient_id, new_data in patient_data.items():
        updated_data = data_updater.update_patient_data(patient_id, new_data)
        data_writer.write_patient_data(patient_id, updated_data)

    print("Update complete.")

if __name__ == "__main__":
    main()