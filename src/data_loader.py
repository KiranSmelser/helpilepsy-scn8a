import pandas as pd
from src import config

def load_weekly_data(file_path):
    """Reads weekly Excel export and returns data split by sheet and patient."""
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    patient_data = {}

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        rename_cols = {
            "Patient Email": "id",
            "Patient Name": "first_name",
            "Patient Surname": "last_name"
        }
        df = df.rename(columns={col: rename_cols[col] for col in rename_cols if col in df.columns})

        # remove Doctor Email column from sheet
        if "Doctor Email" in df.columns:
            df = df.drop(columns=["Doctor Email"])

        # correctly handle dates
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

        if sheet_name.lower() == "medication data":
            med_rename = {
                "Medication Name": "med_name",
                "Reason": "med_reason",
                "Treatment Type": "treatment_type",
                "Intake Type": "med_intake_type"
            }
            df = df.rename(columns={col: med_rename[col] for col in med_rename if col in df.columns})

        if sheet_name.lower() == "seizure data":
            seizure_rename = {
                "Date Seizure": "date_seizure",
                "Date Entry": "date_entry",
                "Seizure Type - HCP": "sz_type_hcp",
                "Seizure Type - Patient": "sz_type_patient",
                "Felt": "felt",
                "During Sleep": "during_sleep",
                "Duration": "duration",
                "Triggers": "triggers",
                "Post Seizure": "post_sz",
                "Emergency Medication": "emergency_med",
                "Auras": "auras",
                "Location": "location",
                "Remarks": "remarks"
            }
            df = df.rename(columns={col: seizure_rename[col] for col in seizure_rename if col in df.columns})

            if "felt" in df.columns:
                df["felt"] = df["felt"].map({"yes": 1, "no": 0})
            if "during_sleep" in df.columns:
                df["during_sleep"] = df["during_sleep"].map({"yes": 1, "no": 0})
            if "duration" in df.columns:
                df["duration"] = pd.to_numeric(df["duration"], errors="coerce").astype("Int64")

        # split Mood & Sleep Data by Event Type column
        if sheet_name.lower() == "mood & sleep data" and "Event Type" in df.columns:
            for event_type, sub_df in df.groupby("Event Type"):
                for pid, group in sub_df.groupby("id"):
                    if pid not in patient_data:
                        patient_data[pid] = {}
                    key = f"{event_type}_data"
                    group = group.drop(columns=["Event Type"])
                    if event_type.lower() in {"mood", "sleep"}:
                        rename_cols = {"Date": "date", "Value": "value"}
                        group = group.rename(columns={col: rename_cols[col] for col in rename_cols if col in group.columns})
                        if "value" in group.columns:
                            group["value"] = pd.to_numeric(group["value"], errors="coerce")
                    patient_data[pid][key] = group.reset_index(drop=True)
        else:
            for pid, group in df.groupby("id"):
                if pid not in patient_data:
                    patient_data[pid] = {}
                patient_data[pid][sheet_name] = group.reset_index(drop=True)

    return patient_data