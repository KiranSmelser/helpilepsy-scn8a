import pandas as pd

def load_excel_data(filepath):
    xls = pd.ExcelFile(filepath)
    seizure_df = xls.parse('Seizure Data')
    medication_df = xls.parse('Medication Data')
    mood_sleep_df = xls.parse('Mood & Sleep Data')
    return seizure_df, medication_df, mood_sleep_df


def clean_seizure_data(df):
    df = df.copy()

    # Keep the original format of 'Date Entry' as string (no conversion to datetime)
    df['Date Entry'] = df['Date Entry'].astype(str).str.strip()

    # Rename columns and use 'Date Entry' as the event date
    df = df.rename(columns={
        'Patient Email': 'patient_email',
        'Patient Name': 'patient_name',
        'Seizure Type - Patient': 'seizure_type',
        'During Sleep': 'during_sleep',
        'Duration': 'duration_sec',
        'Triggers': 'triggers',
        'Post Seizure': 'post_seizure_effects',
        'Location': 'location',
        'Remarks': 'remarks',
        'Date Entry': 'event_date'  # Renaming Date Entry to 'event_date'
    })

    # Diagnostics
    print("First 5 seizure event dates:\n", df['event_date'].head())
    print(f"Missing event dates: {df['event_date'].isna().sum()} out of {len(df)} rows")

    return df[['patient_email', 'patient_name', 'event_date', 'seizure_type',
               'during_sleep', 'duration_sec', 'triggers',
               'post_seizure_effects', 'location', 'remarks']]


def clean_medication_data(df):
    df = df.copy()

    df = df.rename(columns={
        'Patient Email': 'patient_email',
        'Patient Name': 'patient_name',
        'Medication Name': 'medication_name',
        'Reason': 'reason',
        'Treatment Type': 'treatment_type',
        'Intake Type': 'intake_type'
    })

    return df[['patient_email', 'patient_name', 'medication_name',
               'reason', 'treatment_type', 'intake_type']]


def split_mood_sleep_data(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df = df.rename(columns={
        'Patient Email': 'patient_email',
        'Patient Name': 'patient_name',
        'Event Type': 'event_type',
        'Value': 'value'
    })

    mood_df = df[df['event_type'] == 'mood'][['patient_email', 'patient_name', 'Date', 'value']]
    sleep_df = df[df['event_type'] == 'sleep'][['patient_email', 'patient_name', 'Date', 'value']]

    return mood_df, sleep_df
