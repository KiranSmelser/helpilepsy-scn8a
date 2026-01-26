"""Generate per-patient summary statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Import config for lookback window
from . import config
from .diary import (
    extract_med_adherence_diary_entries,
    extract_mood_diary_entries,
    extract_sleep_diary_entries,
)


def _load_if_exists(path: Path) -> pd.DataFrame:
    """Return an empty DataFrame when *path* does not exist or is unreadable."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        tqdm.write(f"[warn] Could not read {path}; assuming empty.")
    return pd.DataFrame()


def _first_created_at(
    df: pd.DataFrame, *, patient_col: str = "patient_id", created_col: str = "createdAt"
) -> pd.Series:
    """Return the earliest UTC `created_col` per patient when available."""
    if df is None or df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    if not {patient_col, created_col}.issubset(df.columns):
        return pd.Series(dtype="datetime64[ns, UTC]")

    tmp = df[[patient_col, created_col]].copy()
    tmp = tmp.dropna(subset=[patient_col, created_col])
    if tmp.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    tmp[created_col] = pd.to_datetime(tmp[created_col], utc=True, errors="coerce")
    tmp = tmp[tmp[created_col].notna()]
    if tmp.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    tmp[patient_col] = tmp[patient_col].astype(str)
    return tmp.groupby(patient_col)[created_col].min()


def compute_metrics(
    processed_dir: Path, lookback_days: int = config.METRICS_LOOKBACK_DAYS
) -> pd.DataFrame:  # noqa: C901
    """Compute average mood, average sleep hours, and seizure history summaries
    for every patient represented in *processed_dir*."""
    processed_dir = Path(processed_dir)

    # Determine cutoff date for lookback window
    today = pd.Timestamp("today").normalize()
    cutoff = today - pd.Timedelta(days=lookback_days)
    # Localize cutoff to UTC for timezone-aware comparisons
    utc_cutoff = cutoff.tz_localize("UTC")

    forms_df = _load_if_exists(processed_dir / "forms.csv")
    form_answers_df = _load_if_exists(processed_dir / "form_answers.csv")
    sleep_diary_entries = extract_sleep_diary_entries(forms_df, form_answers_df)
    mood_diary_entries = extract_mood_diary_entries(forms_df, form_answers_df)
    med_adherence_diary_entries = extract_med_adherence_diary_entries(
        forms_df, form_answers_df
    )

    # Mood/Sleep
    ms_raw = _load_if_exists(processed_dir / "mood_sleep.csv")

    mood_series = pd.Series(dtype=float, name="avg_mood")
    sleep_series = pd.Series(dtype=float, name="avg_sleep_hours")
    sleep_loggers_recent: set[str] = set()
    mood_loggers_recent: set[str] = set()
    ms_df = pd.DataFrame()
    ms_required = {"patient_id", "type", "value"}
    if not ms_raw.empty and ms_required.issubset(ms_raw.columns):
        ms_all = ms_raw.copy()
        ms_all = ms_all[ms_all["patient_id"].notna()].copy()
        if not ms_all.empty:
            ms_all["patient_id"] = ms_all["patient_id"].astype(str)

            ms_recent = ms_all
            if "date" in ms_recent.columns:
                ms_recent["date"] = pd.to_datetime(ms_recent["date"], errors="coerce")
                ms_recent = ms_recent[ms_recent["date"].notna()]
                ms_recent = ms_recent[ms_recent["date"] >= cutoff]
            else:
                ms_recent = pd.DataFrame()

            if not ms_recent.empty:
                sleep_recent = ms_recent.loc[ms_recent["type"] == "sleep", "patient_id"]
                if not sleep_recent.empty:
                    sleep_loggers_recent = set(sleep_recent.unique())
                mood_recent = ms_recent.loc[ms_recent["type"] == "mood", "patient_id"]
                if not mood_recent.empty:
                    mood_loggers_recent = set(mood_recent.unique())

            ms_df = ms_recent

    if not ms_df.empty:
        # ensure numeric values
        ms_df["value_num"] = pd.to_numeric(ms_df["value"], errors="coerce")

        # average mood score
        mood = (
            ms_df.loc[ms_df["type"] == "mood"]
            .groupby("patient_id")["value_num"]
            .mean()
            .rename("avg_mood")
        )
        mood_series = mood

        # average sleep score
        sleep = (
            ms_df.loc[ms_df["type"] == "sleep"]
            .groupby("patient_id")["value_num"]
            .mean()
            .rename("avg_sleep_hours")
        )
        sleep_series = sleep

    if not sleep_diary_entries.empty:
        diary_recent = sleep_diary_entries[
            sleep_diary_entries["date"] >= cutoff
        ].copy()
        if not diary_recent.empty:
            diary_sleep = (
                diary_recent.groupby("patient_id")["score"]
                .mean()
                .rename("avg_sleep_hours")
            )
            if sleep_loggers_recent:
                diary_sleep = diary_sleep[
                    ~diary_sleep.index.astype(str).isin(sleep_loggers_recent)
                ]
            if not diary_sleep.empty:
                if sleep_series.empty:
                    sleep_series = diary_sleep
                else:
                    sleep_series = sleep_series.combine_first(diary_sleep)

    if not mood_diary_entries.empty:
        diary_recent = mood_diary_entries[mood_diary_entries["date"] >= cutoff].copy()
        if not diary_recent.empty:
            diary_mood = (
                diary_recent.groupby("patient_id")["score"]
                .mean()
                .rename("avg_mood")
            )
            if mood_loggers_recent:
                diary_mood = diary_mood[
                    ~diary_mood.index.astype(str).isin(mood_loggers_recent)
                ]
            if not diary_mood.empty:
                if mood_series.empty:
                    mood_series = diary_mood
                else:
                    mood_series = mood_series.combine_first(diary_mood)

    # Medication adherence
    adherence_series = pd.Series(dtype=object).rename("med_adherence")
    diary_adherence_series = pd.Series(
        dtype=float, name="med_adherence_diary_pct"
    )

    mi_path = processed_dir / "med_intakes.csv"
    mi_df = _load_if_exists(mi_path)
    if not mi_df.empty:
        mi = mi_df.copy()
        # Parse dates and filter to lookback window
        mi["date"] = pd.to_datetime(mi["date"], utc=True, errors="coerce")
        mi = mi[mi["date"] >= utc_cutoff]
        mi = mi[mi["date"].notna()]

        # Exclude soft-deleted reminder instances when present
        if "deleted" in mi.columns:
            del_col = mi.get("deleted")
            if del_col is not None:
                del_flag = (
                    del_col.astype(str)
                    .str.lower()
                    .isin(["true", "1", "yes", "y"])
                    .astype(bool)
                )
                mi = mi[~del_flag]

        # Keep rows with valid identifiers
        required_cols = ["patient_id", "medication_id", "intake_id"]
        for c in required_cols:
            if c not in mi.columns:
                mi[c] = None
        mi = mi.dropna(subset=["patient_id", "medication_id", "intake_id"])

        if not mi.empty:
            # Coerce taken flag to boolean
            taken_series = mi.get("taken")
            if taken_series is not None:
                taken_flag = (
                    taken_series.astype(str).str.lower().isin(["true", "1", "yes", "y"])
                )
            else:
                taken_flag = pd.Series([False] * len(mi), index=mi.index)
            mi["taken_flag"] = taken_flag.astype(bool)

            # If dosage index is null use sentinel -1
            mi["dosage_index"] = pd.to_numeric(mi.get("dosage_index"), errors="coerce")
            mi["_dosage_idx_grp"] = mi["dosage_index"].fillna(-1).astype(int)

            group_cols = [
                "patient_id",
                "medication_id",
                "intake_id",
                "_dosage_idx_grp",
                "moment",
            ]
            agg = (
                mi.groupby(group_cols)
                .agg(
                    scheduled_count=("intake_event_id", "count"),
                    taken_count=("taken_flag", "sum"),
                    dose=("dose", "first"),
                    unit=("unit", "first"),
                    intake_from=("intake_from", "first"),
                    intake_to=("intake_to", "first"),
                )
                .reset_index()
                .rename(columns={"_dosage_idx_grp": "dosage_index"})
            )
            if not agg.empty:
                agg["adherence_pct"] = (
                    (agg["taken_count"] / agg["scheduled_count"]) * 100.0
                ).round(1)
                # restore null for missing dosage index
                agg["dosage_index"] = agg["dosage_index"].replace({-1: pd.NA})

                # Enrich with medication name
                med_tbl = _load_if_exists(processed_dir / "medications.csv")
                if not med_tbl.empty and "name" in med_tbl.columns:
                    med_tbl = med_tbl[["medication_id", "name"]].drop_duplicates()
                    agg = agg.merge(
                        med_tbl, how="left", on="medication_id", suffixes=(None, None)
                    )
                    agg = agg.rename(columns={"name": "medication_name"})

                # Window metadata
                window_start = cutoff.date().isoformat()
                window_end = pd.Timestamp("today").normalize().date().isoformat()

                # Build per-patient lists of dicts
                records = []
                for row in agg.to_dict(orient="records"):
                    rec = {
                        "medication_id": row.get("medication_id"),
                        "medication_name": row.get("medication_name"),
                        "intake_id": row.get("intake_id"),
                        "dosage_index": row.get("dosage_index"),
                        "moment": row.get("moment"),
                        "dose": row.get("dose"),
                        "unit": row.get("unit"),
                        "scheduled_count": int(row.get("scheduled_count", 0) or 0),
                        "taken_count": int(row.get("taken_count", 0) or 0),
                        "adherence_pct": float(row.get("adherence_pct", 0.0) or 0.0),
                        "intake_from": row.get("intake_from"),
                        "intake_to": row.get("intake_to"),
                        "window_start": window_start,
                        "window_end": window_end,
                    }
                    rec["patient_id"] = row.get("patient_id")
                    records.append(rec)

                # Pack into a patient-indexed Series of lists
                by_patient: dict[str, list[dict]] = {}
                for r in records:
                    pid = r.pop("patient_id", None)
                    if pid is None:
                        continue
                    by_patient.setdefault(pid, []).append(r)
                adherence_series = pd.Series(by_patient, name="med_adherence")

    if not med_adherence_diary_entries.empty:
        diary_recent = med_adherence_diary_entries.copy()
        diary_recent["date"] = pd.to_datetime(diary_recent["date"], errors="coerce")
        diary_recent = diary_recent[diary_recent["date"].notna()]
        diary_recent = diary_recent[diary_recent["date"] >= cutoff]
        if not diary_recent.empty:
            diary_recent = diary_recent[diary_recent["patient_id"].notna()].copy()
            diary_recent["patient_id"] = diary_recent["patient_id"].astype(str)
            if not diary_recent.empty:
                med_diary = (
                    diary_recent.groupby("patient_id")["score"]
                    .mean()
                    .rename("med_adherence_diary_pct")
                )
                diary_adherence_series = med_diary

    # Seizure history
    events_df = _load_if_exists(processed_dir / "events.csv")

    seizure_dates_series = pd.Series(dtype=object)
    longest_free_series = pd.Series(dtype=float)

    if not events_df.empty:
        seiz_df = events_df.loc[events_df["type"] == "seizure"].copy()
        # Parse dates and filter to lookback window
        seiz_df["date"] = pd.to_datetime(seiz_df["date"], utc=True, errors="coerce")
        utc_cutoff = cutoff.tz_localize("UTC")
        seiz_df = seiz_df[seiz_df["date"] >= utc_cutoff]
        seiz_df = seiz_df[seiz_df["date"].notna()]
        if not seiz_df.empty:
            # Work with midnight‑normalized timestamps
            seiz_df["date_only"] = seiz_df["date"].dt.normalize().dt.tz_localize(None)

            grouped = seiz_df.groupby("patient_id")["date_only"]

            # List of unique seizure dates per patient
            seizure_dates_series = grouped.apply(
                lambda s: sorted({d.date().isoformat() for d in s})
            ).rename("seizure_dates")

            today = pd.Timestamp("today").normalize()

            def _longest_gap(dates: list[pd.Timestamp]) -> int | None:
                """Return the maximum seizure-free run (in days).

                We count only seizure-free days *between* seizure events and the
                seizure-free days since the latest seizure up to *today*.
                """
                if not dates:
                    return None

                dates_sorted = sorted(dates)
                longest = 0

                for prev, cur in zip(dates_sorted, dates_sorted[1:]):
                    gap = (cur.normalize() - prev.normalize()).days - 1
                    longest = max(longest, max(0, gap))

                since_last = (today - dates_sorted[-1].normalize()).days
                longest = max(longest, max(0, since_last))
                return int(longest)

            longest_free_series = grouped.apply(lambda s: _longest_gap(list(s))).rename(
                "longest_seizure_free_days"
            )

    # App usage

    usage_days_series = pd.Series(dtype="int64").rename("app_usage_days_total")
    usage_latest_series = pd.Series(dtype="int64").rename("app_usage_days_latest_month")
    usage_per_month_series = pd.Series(dtype="object").rename("app_usage_per_month")

    usage_parts: list[pd.DataFrame] = []

    # Define a strict date window of exactly lookback_days calendar days
    window_start = (cutoff + pd.Timedelta(days=1)).normalize()
    window_end = today  # inclusive

    # Collect usage dates from events
    if not events_df.empty:
        ev = events_df.copy()
        ev["date"] = pd.to_datetime(ev["date"], utc=True, errors="coerce")
        ev = ev[ev["date"].notna()]
        if not ev.empty:
            ev["date_only"] = ev["date"].dt.normalize().dt.tz_localize(None)
            ev = ev[(ev["date_only"] >= window_start) & (ev["date_only"] <= window_end)]
            if not ev.empty:
                usage_parts.append(ev[["patient_id", "date_only"]])

    # Collect usage dates from forms
    if not forms_df.empty:
        fr = forms_df.copy()
        fr["date"] = pd.to_datetime(fr["date"], utc=True, errors="coerce")
        fr = fr[fr["date"].notna()]
        if not fr.empty:
            fr["date_only"] = fr["date"].dt.normalize().dt.tz_localize(None)
            fr = fr[(fr["date_only"] >= window_start) & (fr["date_only"] <= window_end)]
            if not fr.empty:
                usage_parts.append(fr[["patient_id", "date_only"]])

    # Collect usage dates from medication intake confirmations
    mi_usage = _load_if_exists(processed_dir / "med_intakes.csv")
    if not mi_usage.empty:
        mi = mi_usage.copy()
        taken_col = mi.get("taken")
        if taken_col is not None:
            taken_flag = (
                taken_col.astype(str)
                .str.lower()
                .isin(["true", "1", "yes", "y"])
                .astype(bool)
            )
        else:
            taken_flag = pd.Series([False] * len(mi), index=mi.index)
        mi["taken_date"] = pd.to_datetime(
            mi.get("taken_date"), utc=True, errors="coerce"
        )
        mi = mi[taken_flag & mi["taken_date"].notna()]
        if not mi.empty:
            mi["date_only"] = mi["taken_date"].dt.normalize().dt.tz_localize(None)
            mi = mi[(mi["date_only"] >= window_start) & (mi["date_only"] <= window_end)]
            if not mi.empty:
                usage_parts.append(mi[["patient_id", "date_only"]])

    if usage_parts:
        usage_df = pd.concat(usage_parts, ignore_index=True)
        usage_df = usage_df.dropna(subset=["patient_id", "date_only"])
        # De-duplicate so multiple interactions on the same day only count once
        usage_df = usage_df.drop_duplicates()

        usage_days_series = (
            usage_df.groupby("patient_id").size().rename("app_usage_days_total")
        )

        # Per-month counts
        usage_df["month"] = usage_df["date_only"].dt.to_period("M").astype(str)
        monthly_counts = (
            usage_df.groupby(["patient_id", "month"])
            .size()
            .rename("count")
            .reset_index()
        )

        usage_per_month_series = (
            monthly_counts.sort_values(["patient_id", "month"])
            .groupby("patient_id")[["month", "count"]]
            .apply(lambda df: dict(zip(df["month"], df["count"])))
            .rename("app_usage_per_month")
        )

        # Latest-month count
        idx = monthly_counts.groupby("patient_id")["month"].idxmax()
        usage_latest_series = (
            monthly_counts.loc[idx]
            .set_index("patient_id")["count"]
            .astype(int)
            .rename("app_usage_days_latest_month")
        )

    # Prospective survey completion
    prospective_series = pd.Series(
        dtype=bool, name="prospective_survey_completed"
    )
    prospective_completed_at_series = pd.Series(
        dtype=object, name="prospective_survey_completed_at"
    )
    ps_path = processed_dir / "prospective_surveys.csv"
    ps_df = _load_if_exists(ps_path)
    if not ps_df.empty and {
        "patient_id",
        "prospective_study_complete",
    }.issubset(ps_df.columns):
        ps = ps_df.loc[ps_df["patient_id"].notna()].copy()
        if not ps.empty:
            ps["patient_id"] = ps["patient_id"].astype(str)
            completion_tokens = (
                ps["prospective_study_complete"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            ps["_prospective_complete"] = completion_tokens.isin({"2", "complete"})
            complete_by_patient = (
                ps.groupby("patient_id")["_prospective_complete"]
                .max()
                .astype(bool)
            )
            prospective_series = complete_by_patient.rename(
                "prospective_survey_completed"
            )
            timestamp_field = next(
                (
                    col
                    for col in (
                        "prospective_study_timestamp_utc",
                        "prospective_study_timestamp",
                    )
                    if col in ps.columns
                ),
                None,
            )
            if timestamp_field:
                parsed_ts = pd.to_datetime(
                    ps[timestamp_field],
                    utc=True,
                    errors="coerce",
                )
                ps["_prospective_ts"] = parsed_ts
                completed_ts = ps.loc[
                    ps["_prospective_complete"] & ps["_prospective_ts"].notna(),
                    ["patient_id", "_prospective_ts"],
                ]
                if not completed_ts.empty:
                    latest_ts = (
                        completed_ts.groupby("patient_id")["_prospective_ts"]
                        .max()
                        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        .rename("prospective_survey_completed_at")
                    )
                    prospective_completed_at_series = latest_ts

    # First app activity
    first_activity_series = pd.Series(
        dtype=object, name="first_app_activity_createdAt"
    )
    candidate_firsts: list[pd.Series] = []
    if not events_df.empty and {
        "patient_id",
        "type",
        "createdAt",
    }.issubset(events_df.columns):
        patient_event_types = {"seizure", "side_effect", "headache", "other", "form"}
        ev_subset = events_df.loc[events_df["type"].isin(patient_event_types)].copy()
        first_events = _first_created_at(ev_subset)
        if not first_events.empty:
            candidate_firsts.append(first_events)

    if not forms_df.empty:
        first_forms = _first_created_at(forms_df)
        if not first_forms.empty:
            candidate_firsts.append(first_forms)

    meds_df = _load_if_exists(processed_dir / "medications.csv")
    if not meds_df.empty:
        first_meds = _first_created_at(meds_df)
        if not first_meds.empty:
            candidate_firsts.append(first_meds)

    if candidate_firsts:
        merged_first = pd.concat(candidate_firsts, axis=1)
        min_ts = merged_first.min(axis=1, skipna=True)
        min_ts = min_ts[min_ts.notna()]
        if not min_ts.empty:
            tz_info = getattr(min_ts.dt, "tz", None)
            if tz_info is None:
                min_ts = min_ts.dt.tz_localize("UTC")
            else:
                min_ts = min_ts.dt.tz_convert("UTC")
            first_activity_series = min_ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").rename(
                "first_app_activity_createdAt"
            )

    # WhatsApp group membership
    whatsapp_series = pd.Series(dtype="bool", name="whatsapp_group_member")
    whatsapp_path = processed_dir / "whatsapp_status.csv"
    whatsapp_df = _load_if_exists(whatsapp_path)
    if not whatsapp_df.empty and {"patient_id", "whatsapp"}.issubset(whatsapp_df.columns):
        wa = whatsapp_df[["patient_id", "whatsapp"]].dropna(subset=["patient_id"]).copy()
        if not wa.empty:
            wa["patient_id"] = wa["patient_id"].astype(str)
            wa["whatsapp_flag"] = (
                pd.to_numeric(wa["whatsapp"], errors="coerce")
                .fillna(0)
                .clip(lower=0, upper=1)
                .astype(int)
            )
            whatsapp_series = (
                wa.groupby("patient_id")["whatsapp_flag"]
                .max()
                .astype(bool)
                .rename("whatsapp_group_member")
            )

    # Combine
    metrics = (
        pd.concat(
            [
                mood_series,
                sleep_series,
                adherence_series,
                diary_adherence_series,
                seizure_dates_series,
                longest_free_series,
                usage_days_series,
                usage_latest_series,
                prospective_series,
                prospective_completed_at_series,
                usage_per_month_series,
                first_activity_series,
                whatsapp_series,
            ],
            axis=1,
        )
        .reset_index(names="patient_id")
        .sort_values("patient_id")
    )

    return metrics


def generate(processed_dir: Path) -> Path:
    """Public helper invoked by the ETL main.py script."""
    metrics_df = compute_metrics(processed_dir)
    out_path = Path(processed_dir) / "patient_summary_metrics.json"

    # Convert to ordinary Python types for clean JSON serialization
    records = metrics_df.to_dict(orient="records")

    def _sanitize(obj):  # recursively replace NaN/NaT with None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        try:
            if pd.isna(obj):
                return None
        except Exception:  # noqa: BLE001
            pass
        return obj

    clean_records = _sanitize(records)

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(clean_records, fp, indent=2, ensure_ascii=False, allow_nan=False)

    return out_path
