import pandas as pd

def load_intake_excel(filepath):
    """Load the intake confirmations sheet from the Excel file."""
    xls = pd.ExcelFile(filepath)
    print("Available sheets:", xls.sheet_names)
    intake_df = xls.parse('Intake confirmations')
    return intake_df


def clean_intake_confirmations_data(df):
    df = df.copy()

    # Keep 'taken_date' as string to preserve original format
    df['taken_date'] = df['taken_date'].astype(str).str.strip()

    # Rename for clarity and consistency
    df = df.rename(columns={
        'email': 'patient_email',
        'taken_date': 'event_date',
        'name': 'medication_name',
        'intake_type': 'intake_type',
        'reminderDose': 'dose',
        'reminderUnit': 'unit',
        'taken': 'taken'
    })

    # Optional: Print some diagnostics
    print("First 5 event dates:\n", df['event_date'].head())
    print(f"Missing intake dates: {df['event_date'].isna().sum()} of {len(df)} rows")

    return df[['patient_email', 'event_date', 'medication_name', 'intake_type', 'dose', 'unit', 'taken']]
