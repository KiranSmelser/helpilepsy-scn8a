"""Create a summary graphic for every patient."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, DrawingArea
from matplotlib.lines import Line2D
import pandas as pd
from tqdm import tqdm

# Import config for lookback window
from . import config
from .diary import (
    extract_mood_diary_entries,
    extract_sleep_diary_entries,
)


# Helpers


def _format_value(val: Any, fmt: str = "{:.1f}") -> str:
    """Return *val* formatted or an en-dash when null/NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)):  # noqa: PLR1736
        return "—"
    try:
        return fmt.format(int(val))
    except Exception:  # noqa: BLE001
        return str(int(val))


def _build_streak_series(
    seizure_dates: list[str],
) -> tuple[list[date], list[int]]:
    """Build (x, y) for a step-plot of seizure-free days over the lookback window."""
    today = date.today()
    lookback = int(getattr(config, "METRICS_LOOKBACK_DAYS", 30))
    # Strict window of exactly lookback calendar days
    start = today - timedelta(days=lookback - 1)

    # Parse provided seizure dates
    seizure_set: set[date] = set()
    try:
        if seizure_dates:
            seizure_set = {
                datetime.strptime(d, "%Y-%m-%d").date() for d in seizure_dates if d
            }
    except Exception:
        seizure_set = set()

    x: list[date] = []
    y: list[int] = []
    streak = 0
    cur = start
    while cur <= today:
        if cur in seizure_set:
            # Seizure day
            streak = 0
            x.append(cur)
            y.append(0)
        else:
            # Seizure‑free day
            streak += 1
            x.append(cur)
            y.append(streak)
        cur += timedelta(days=1)

    return x, y


# Panel/Background helpers


def _add_rounded_panel(
    ax,
    facecolor: str,
    inset_x: float = 0.0,
    width: float = 1.0,
    pad: float = 0.06,
    round: float = 0.08,
) -> None:
    """Add a rounded rectangle panel behind *ax* contents with visible rounded corners."""
    # Make the Axes background transparent so the rounded patch is visible
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)

    # Draw a rounded rectangle that sits *inside* the axes area
    panel = FancyBboxPatch(
        (inset_x, 0.0),
        width,
        1.0,
        boxstyle=f"round,pad={pad},rounding_size={round}",
        transform=ax.transAxes,
        linewidth=0,
        facecolor=facecolor,
        edgecolor="none",
        clip_on=False,
        zorder=-1,
        antialiased=True,
    )
    ax.add_patch(panel)


# Pill label helper


def _draw_pill(
    ax,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str,
    textcolor: str = "white",
    fontsize: int = 11,
    shadow: bool = True,
) -> None:
    """Draw a capsule-shaped label with optional soft shadow."""
    # subtle drop shadow for depth
    if shadow:
        shadow_patch = FancyBboxPatch(
            (x + 0.004, y - 0.004),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={h/2}",
            transform=ax.transAxes,
            linewidth=0,
            facecolor="black",
            alpha=0.10,
            zorder=2,
            clip_on=False,
            antialiased=True,
        )
        ax.add_patch(shadow_patch)

    # main pill
    pill = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={h/2}",
        transform=ax.transAxes,
        linewidth=0,
        facecolor=facecolor,
        edgecolor="none",
        zorder=3,
        antialiased=True,
    )
    ax.add_patch(pill)

    # centered label text
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=textcolor,
        zorder=4,
    )


# Public API


def generate(processed_dir: Path | str) -> list[Path]:
    """Build summary graphics for every patient."""
    processed_dir = Path(processed_dir)
    metrics_path = processed_dir / "patient_summary_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    with open(metrics_path, "r", encoding="utf-8") as fp:
        records: List[dict[str, Any]] = json.load(fp)

    out_dir = processed_dir / "patient_graphics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify patients who have ever reported a seizure
    prior_seizure_pids: set[Any] = set()
    # Compute ongoing seizure‑free streak
    current_streak_days_by_pid: dict[str, int] = {}
    try:
        events_path = processed_dir / "events.csv"
        if events_path.exists():
            ev = pd.read_csv(events_path)
            if not ev.empty and {"type", "patient_id", "date"}.issubset(ev.columns):
                ev = ev.copy()
                ev["date"] = pd.to_datetime(ev["date"], utc=True, errors="coerce")
                ev = ev[(ev["type"] == "seizure") & ev["date"].notna()]
                # only consider events up to now
                today_utc = pd.Timestamp("today", tz="UTC")
                ev = ev[ev["date"] <= today_utc]
                prior_seizure_pids = set(ev["patient_id"].dropna().unique())

                # Days since last seizure per patient
                if not ev.empty:
                    ev["date_only"] = ev["date"].dt.normalize().dt.tz_localize(None)
                    last_by_pid = ev.groupby("patient_id")["date_only"].max()
                    today_naive = pd.Timestamp("today").normalize()
                    days_since_last = (today_naive - last_by_pid).dt.days.astype(int)
                    # Normalize keys to string for robust lookups later
                    current_streak_days_by_pid = {
                        str(pid): int(days) for pid, days in days_since_last.items()
                    }
    except Exception:
        prior_seizure_pids = set()
        current_streak_days_by_pid = {}

    # Normalize to string keys for membership checks
    prior_seizure_pid_keys = {str(p) for p in prior_seizure_pids}

    written: list[Path] = []
    lookback_days = int(getattr(config, "METRICS_LOOKBACK_DAYS", 30))
    lookback_cutoff = pd.Timestamp("today").normalize() - pd.Timedelta(
        days=lookback_days
    )
    # Precompute which patients have any mood/sleep data ever vs. within lookback
    has_mood_ever: set[str] = set()
    has_mood_in_window: set[str] = set()
    has_sleep_ever: set[str] = set()
    has_sleep_in_window: set[str] = set()
    try:
        ms_path = processed_dir / "mood_sleep.csv"
        if ms_path.exists():
            ms = pd.read_csv(ms_path)
            if not ms.empty and {"type", "patient_id", "date"}.issubset(ms.columns):
                # any mood ever
                mood_all = ms.loc[ms["type"] == "mood"].copy()
                if not mood_all.empty:
                    has_mood_ever = set(
                        mood_all["patient_id"].dropna().astype(str).unique()
                    )

                    # mood within lookback window
                    mood_all["date"] = pd.to_datetime(mood_all["date"], errors="coerce")
                    mood_lb = mood_all[mood_all["date"] >= lookback_cutoff]
                    if not mood_lb.empty:
                        has_mood_in_window = set(
                            mood_lb["patient_id"].dropna().astype(str).unique()
                        )

                # any sleep ever
                sleep_all = ms.loc[ms["type"] == "sleep"].copy()
                if not sleep_all.empty:
                    has_sleep_ever = set(
                        sleep_all["patient_id"].dropna().astype(str).unique()
                    )

                    # sleep within lookback window
                    sleep_all["date"] = pd.to_datetime(
                        sleep_all["date"], errors="coerce"
                    )
                    sleep_lb = sleep_all[sleep_all["date"] >= lookback_cutoff]
                    if not sleep_lb.empty:
                        has_sleep_in_window = set(
                            sleep_lb["patient_id"].dropna().astype(str).unique()
                        )
    except Exception:
        has_mood_ever = set()
        has_mood_in_window = set()
    diary_sleep_entries = pd.DataFrame(columns=["patient_id", "date", "score"])
    diary_mood_entries = pd.DataFrame(columns=["patient_id", "date", "score"])
    try:
        forms_path = processed_dir / "forms.csv"
        answers_path = processed_dir / "form_answers.csv"
        if forms_path.exists() and answers_path.exists():
            forms_df = pd.read_csv(forms_path)
            answers_df = pd.read_csv(answers_path)
            diary_sleep_entries = extract_sleep_diary_entries(forms_df, answers_df)
            diary_mood_entries = extract_mood_diary_entries(forms_df, answers_df)
    except Exception:
        diary_sleep_entries = pd.DataFrame(columns=["patient_id", "date", "score"])
        diary_mood_entries = pd.DataFrame(columns=["patient_id", "date", "score"])

    existing_sleep_loggers = set(has_sleep_ever)
    existing_mood_loggers = set(has_mood_ever)

    if not diary_sleep_entries.empty:
        sleep_diary = diary_sleep_entries.copy()
        sleep_diary["date"] = pd.to_datetime(sleep_diary["date"], errors="coerce")
        sleep_diary = sleep_diary[sleep_diary["date"].notna()]
        sleep_diary["patient_id"] = sleep_diary["patient_id"].astype(str)
        diary_patients = set(sleep_diary["patient_id"].unique())
        eligible_patients = diary_patients - existing_sleep_loggers
        if eligible_patients:
            has_sleep_ever.update(eligible_patients)
            diary_lb = sleep_diary[sleep_diary["date"] >= lookback_cutoff]
            if not diary_lb.empty:
                eligible_lb = (
                    set(diary_lb["patient_id"].unique()) & eligible_patients
                )
                if eligible_lb:
                    has_sleep_in_window.update(eligible_lb)

    if not diary_mood_entries.empty:
        mood_diary = diary_mood_entries.copy()
        mood_diary["date"] = pd.to_datetime(mood_diary["date"], errors="coerce")
        mood_diary = mood_diary[mood_diary["date"].notna()]
        mood_diary["patient_id"] = mood_diary["patient_id"].astype(str)
        diary_patients = set(mood_diary["patient_id"].unique())
        eligible_patients = diary_patients - existing_mood_loggers
        if eligible_patients:
            has_mood_ever.update(eligible_patients)
            diary_lb = mood_diary[mood_diary["date"] >= lookback_cutoff]
            if not diary_lb.empty:
                eligible_lb = (
                    set(diary_lb["patient_id"].unique()) & eligible_patients
                )
                if eligible_lb:
                    has_mood_in_window.update(eligible_lb)
    patient_names: dict[str, str] = {}
    try:
        whatsapp_path = processed_dir / "whatsapp_status.csv"
        if whatsapp_path.exists():
            whats = pd.read_csv(whatsapp_path, dtype=str)
            required = ["patient_id", "first_name", "last_name"]
            if not whats.empty and set(required).issubset(whats.columns):
                subset = whats[required].copy()
                subset["patient_id"] = subset["patient_id"].astype(str)
                subset["first_name"] = (
                    subset["first_name"].fillna("").astype(str).str.strip()
                )
                subset["last_name"] = (
                    subset["last_name"].fillna("").astype(str).str.strip()
                )
                subset["full_name"] = (
                    subset["first_name"] + " " + subset["last_name"]
                ).str.strip()
                subset = subset[subset["full_name"].ne("")]
                patient_names = (
                    subset.drop_duplicates("patient_id")
                    .set_index("patient_id")["full_name"]
                    .to_dict()
                )
    except Exception:
        patient_names = {}

    for rec in tqdm(records, desc="Summary graphics", unit="patient"):
        pid = rec.get("patient_id")
        if pid is None:
            continue

        avg_mood = rec.get("avg_mood")
        # Mood values already stored on a 0–10 scale; clamp to guard bad data
        mood_score: float | None = None
        if avg_mood is not None and not pd.isna(avg_mood):
            mood_score = max(0.0, min(10.0, float(avg_mood)))
        avg_sleep = rec.get("avg_sleep_hours")
        # Sleep values also use the same 0–10 scale
        sleep_score: float | None = None
        if avg_sleep is not None and not pd.isna(avg_sleep):
            sleep_score = max(0.0, min(10.0, float(avg_sleep)))

        # Medication adherence percentage
        med_adherence = rec.get("med_adherence")
        diary_adherence = rec.get("med_adherence_diary_pct")
        adh_pct: float | None = None
        if diary_adherence is not None and not pd.isna(diary_adherence):
            try:
                adh_pct = max(0.0, min(100.0, float(diary_adherence)))
            except Exception:
                adh_pct = None
        if adh_pct is None:
            if isinstance(med_adherence, (int, float)) and not pd.isna(med_adherence):
                adh_pct = float(med_adherence) * 100.0
            elif isinstance(med_adherence, list) and len(med_adherence) > 0:
                try:
                    sched = sum(
                        int(x.get("scheduled_count", 0) or 0) for x in med_adherence
                    )
                    taken = sum(int(x.get("taken_count", 0) or 0) for x in med_adherence)
                    if sched > 0:
                        adh_pct = round((taken / sched) * 100.0, 1)
                except Exception:
                    adh_pct = None

        # Seizure dates list
        seiz_raw = rec.get("seizure_dates")
        seiz_dates: list[str] = seiz_raw if isinstance(seiz_raw, list) else []

        # Longest seizure‑free streak
        longest_gap = rec.get("longest_seizure_free_days")

        # App usage metrics
        usage_total = rec.get("app_usage_days_total")
        usage_latest = rec.get("app_usage_days_latest_month")
        usage_months = rec.get("app_usage_per_month")
        prospective_completed = bool(rec.get("prospective_survey_completed"))
        whatsapp_member = bool(rec.get("whatsapp_group_member"))
        whatsapp_status_label = "Member" if whatsapp_member else "Non-member"
        whatsapp_status_fontsize = 22 if whatsapp_member else 20
        first_activity_raw = rec.get("first_app_activity_createdAt")
        first_activity_ts = None
        if first_activity_raw:
            try:
                first_activity_ts = pd.to_datetime(
                    first_activity_raw, utc=True, errors="coerce"
                )
            except Exception:
                first_activity_ts = None

        prospective_completed_at_raw = rec.get("prospective_survey_completed_at")
        prospective_completed_ts = None
        if prospective_completed_at_raw:
            try:
                prospective_completed_ts = pd.to_datetime(
                    prospective_completed_at_raw, utc=True, errors="coerce"
                )
            except Exception:
                prospective_completed_ts = None

        def _offset_label(base_ts: pd.Timestamp | None, months: int) -> str:
            """Return a dd/mm/YYYY date string offset by *months* from *base_ts*."""
            if base_ts is None or pd.isna(base_ts):
                return "—"
            try:
                target_ts = base_ts + pd.DateOffset(months=months)
                if isinstance(target_ts, pd.Timestamp):
                    if target_ts.tzinfo is not None:
                        target_ts = target_ts.tz_convert("UTC")
                    return target_ts.date().strftime("%d/%m/%Y")
                # Fallback when DateOffset returns datetime-like object
                target = pd.to_datetime(target_ts, utc=True, errors="coerce")
                if target is not None and not pd.isna(target):
                    return target.date().strftime("%d/%m/%Y")
            except Exception:
                pass
            return "—"

        # Determine upcoming/final survey labels based on completion status
        if prospective_completed:
            base_ts = prospective_completed_ts or first_activity_ts
            upcoming_survey_label = _offset_label(base_ts, 6)
            final_survey_label = _offset_label(base_ts, 12)
        else:
            # Hide future survey dates until the current survey is finished
            upcoming_survey_label = "—"
            final_survey_label = "—"

        # Figure scaffold
        fig = plt.figure(figsize=(10, 9), dpi=300)
        # Background
        fig.patch.set_facecolor("#FFF7EB")
        display_name = patient_names.get(str(pid))
        # 3x4 grid layout
        grid = fig.add_gridspec(
            3,
            4,
            hspace=0.2,
            wspace=0.22,
            top=0.90,
            height_ratios=[1.0, 1.0, 1.0],
            width_ratios=[1.0, 1.0, 1.0, 1.0],
        )

        # Mood
        ax_mood = fig.add_subplot(grid[1, 0])
        ax_mood.axis("off")
        _add_rounded_panel(ax_mood, "#FFE7C6")
        # Pill-style label
        _draw_pill(
            ax_mood,
            text="Mood",
            x=0.015,
            y=0.88,
            w=0.32,
            h=0.12,
            facecolor="#E88F2D",
            fontsize=12,
        )

        # Render descriptor + value or no-data message
        if mood_score is not None:
            ax_mood.text(
                0.02,
                0.35,
                "Average\nmood",
                transform=ax_mood.transAxes,
                fontsize=12,
                va="top",
                ha="left",
                color="#D26A26",
            )
            ax_mood.text(
                0.02,
                0.000001,
                _format_value(mood_score, "{:.0f}/10"),
                transform=ax_mood.transAxes,
                fontsize=30,
                fontweight="bold",
                va="bottom",
                ha="left",
                color="#D26A26",
            )
        else:
            pid_str = str(pid)
            if pid_str in has_mood_ever and pid_str not in has_mood_in_window:
                msg = f"No mood data in the\nlast {int(getattr(config, 'METRICS_LOOKBACK_DAYS', 30))} days"
            else:
                msg = "No mood data"
            ax_mood.text(
                0.02,
                0.30,
                msg,
                transform=ax_mood.transAxes,
                fontsize=12,
                fontstyle="italic",
                va="bottom",
                ha="left",
                color="#D26A26",
            )

        # Add mood icon based on mood score
        try:
            # Determine appropriate face image
            face_file = "face_4.png"
            if mood_score is not None:
                # Round to nearest integer
                mood_int = int(mood_score)
                if mood_int <= 3:
                    face_file = "face_0.png"
                elif 4 <= mood_int <= 6:
                    face_file = "face_1.png"
                elif 7 <= mood_int <= 8:
                    face_file = "face_2.png"
                elif mood_int >= 9:
                    face_file = "face_3.png"
            icon_path = processed_dir.parent.parent / "img" / face_file
            if icon_path.exists():
                arr = plt.imread(icon_path)
                imagebox = OffsetImage(arr, zoom=0.12)
                ab = AnnotationBbox(
                    imagebox,
                    (0.68, 0.74),
                    frameon=False,
                    xycoords="axes fraction",
                    zorder=2,
                )
                ax_mood.add_artist(ab)
        except Exception:
            pass

        # Sleep
        ax_sleep = fig.add_subplot(grid[1, 1])
        ax_sleep.axis("off")
        _add_rounded_panel(ax_sleep, "#C6E7F7")
        # Pill-style label
        _draw_pill(
            ax_sleep,
            text="Sleep",
            x=0.015,
            y=0.88,
            w=0.32,
            h=0.12,
            facecolor="#3A8FD8",
            fontsize=12,
        )

        # Render descriptor + value or no-data message
        if sleep_score is not None:
            ax_sleep.text(
                0.02,
                0.35,
                "Average\nsleep",
                transform=ax_sleep.transAxes,
                fontsize=12,
                va="top",
                ha="left",
                color="#174E80",
            )
            ax_sleep.text(
                0.02,
                0.000001,
                _format_value(sleep_score, "{:.0f}/10"),
                transform=ax_sleep.transAxes,
                fontsize=30,
                fontweight="bold",
                va="bottom",
                ha="left",
                color="#174E80",
            )
        else:
            pid_str = str(pid)
            if pid_str in has_sleep_ever and pid_str not in has_sleep_in_window:
                msg = f"No sleep data in the\nlast {int(getattr(config, 'METRICS_LOOKBACK_DAYS', 30))} days"
            else:
                msg = "No sleep data"
            ax_sleep.text(
                0.02,
                0.30,
                msg,
                transform=ax_sleep.transAxes,
                fontsize=12,
                fontstyle="italic",
                va="bottom",
                ha="left",
                color="#174E80",
            )

        # Add moon phase icon based on sleep score
        try:
            phase_file = "phase_0.png"
            if sleep_score is not None:
                # Round to nearest integer
                sleep_int = int(sleep_score)
                if sleep_int == 0:
                    phase_file = "phase_0.png"
                elif 1 <= sleep_int <= 2:
                    phase_file = "phase_1.png"
                elif 3 <= sleep_int <= 4:
                    phase_file = "phase_2.png"
                elif 5 <= sleep_int <= 6:
                    phase_file = "phase_3.png"
                elif 7 <= sleep_int <= 8:
                    phase_file = "phase_4.png"
                elif sleep_int >= 9:
                    phase_file = "phase_5.png"
            icon_path = processed_dir.parent.parent / "img" / phase_file
            if icon_path.exists():
                arr = plt.imread(icon_path)
                imagebox = OffsetImage(arr, zoom=0.15)
                # Raise moon icon when sleep score is 10
                phase_xy = (0.79, 0.18)
                if sleep_score is not None and int(sleep_score) == 10:
                    phase_xy = (0.79, 0.32)
                ab = AnnotationBbox(
                    imagebox,
                    phase_xy,
                    frameon=False,
                    xycoords="axes fraction",
                    zorder=2,
                )
                ax_sleep.add_artist(ab)
        except Exception:
            pass

        # Add sleep icon in sleep panel
        try:
            icon_path = processed_dir.parent.parent / "img" / "sleep.png"
            if icon_path.exists():
                arr = plt.imread(icon_path)
                imagebox = OffsetImage(arr, zoom=0.17)
                ab = AnnotationBbox(
                    imagebox,
                    (0.68, 0.76),
                    frameon=False,
                    xycoords="axes fraction",
                    zorder=2,
                )
                ax_sleep.add_artist(ab)
        except Exception:
            pass

        # Seizure‑free streak
        ax_streak = fig.add_subplot(grid[0, :])
        if display_name:
            bbox = ax_streak.get_position()
            streak_top = min(1.0, bbox.y1)
            name_y = streak_top + (1.0 - streak_top) * 0.5
            fig.text(
                0.5,
                name_y,
                display_name,
                fontsize=24,
                fontweight="bold",
                va="center",
                ha="center",
                color="#2F2F2F",
            )
        _add_rounded_panel(ax_streak, "#D2F1D8", 0.05, 0.9, round=0.03)
        # Title
        ax_streak.text(
            0.5,
            0.995,
            "Seizure-Free Streak",
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="center",
            color="#59B37B",
        )

        # Plot streak
        has_prior_seizure = str(pid) in prior_seizure_pid_keys
        if seiz_dates or has_prior_seizure:
            x, y = _build_streak_series(seiz_dates)

            # Hide outer axes spines
            for spine in ax_streak.spines.values():
                spine.set_visible(False)
            # Remove outer ticks/labels
            ax_streak.set_xticks([])
            ax_streak.set_yticks([])
            ax_streak.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Create an inset axes inside the seizure panel
            plot_ax = ax_streak.inset_axes([0.06, 0.10, 0.90, 0.62])
            plot_ax.set_facecolor("none")
            plot_ax.patch.set_alpha(0.0)

            # Step curve
            plot_ax.step(
                x,
                y,
                where="post",
                color="#59B37B",
                linewidth=2.6,
                solid_capstyle="round",
                zorder=3,
            )
            # Mark seizure events along x‑axis
            filtered_seiz = [d for d in seiz_dates if d]
            if filtered_seiz:
                plot_ax.scatter(
                    [datetime.strptime(d, "%Y-%m-%d") for d in filtered_seiz],
                    [0] * len(filtered_seiz),
                    s=26,
                    zorder=5,
                    facecolor="#FF7A66",
                    edgecolor="none",
                    alpha=0.95,
                    clip_on=False,
                )

            # Remove box outline and label months on x-axis
            for spine in plot_ax.spines.values():
                spine.set_visible(False)
            plot_ax.set_facecolor("none")
            plot_ax.patch.set_alpha(0.0)
            # Show y-axis tick labels
            plot_ax.set_ylabel("Days", labelpad=10, fontsize=8)
            plot_ax.tick_params(
                axis="y", left=False, labelleft=True, labelsize=8, length=0
            )
            plot_ax.tick_params(axis="x", bottom=True, labelbottom=True)

            plot_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plot_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            plot_ax.minorticks_off()
            plot_ax.tick_params(axis="x", rotation=0, labelsize=8, length=0)

            # Horizontal gridlines
            plot_ax.set_axisbelow(True)
            # Ensure y-range
            y_max = max(y) if y else 0
            y_top = max(30, y_max + 1)
            plot_ax.set_ylim(0, y_top)
            plot_ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            plot_ax.grid(axis="y", linestyle="-", alpha=0.15)

            # Legend
            legend_handles = [
                Line2D([0], [0], color="#59B37B", linewidth=2.6, label="Seizure-free")
            ]
            if filtered_seiz:
                legend_handles.insert(
                    0,
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markerfacecolor="#FF7A66",
                        markeredgecolor="none",
                        markersize=6,
                        label="Seizure event(s)",
                    ),
                )
            plot_ax.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=2,
                frameon=False,
                fontsize=8,
                handletextpad=0.8,
                columnspacing=1.0,
                borderaxespad=0.0,
            )
        else:
            ax_streak.axis("off")
            ax_streak.text(
                0.5,
                0.5,
                "No seizure data",
                ha="center",
                va="center",
                fontsize=9,
                fontstyle="italic",
            )

        # Longest streak annotation
        da = DrawingArea(16, 16, 0, 0)
        circ = Circle((8, 8), 8, facecolor="#59B37B", edgecolor="none")
        da.add_artist(circ)
        ab_circ = AnnotationBbox(
            da,
            (0.08, 0.82),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=5,
        )
        ax_streak.add_artist(ab_circ)

        ax_streak.text(
            0.08,
            0.815,
            "\u2713",
            transform=ax_streak.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
            zorder=6,
        )

        no_in_window_seizures = not seiz_dates
        if no_in_window_seizures and has_prior_seizure:
            ongoing = current_streak_days_by_pid.get(str(pid))
            if ongoing is not None:
                longest_text_val = f"{int(ongoing)}"
            else:
                longest_text_val = _format_value(longest_gap, "{:.0f}")
        else:
            longest_text_val = _format_value(longest_gap, "{:.0f}")

        ax_streak.text(
            0.10,
            0.805,
            f"Longest streak: {longest_text_val} days",
            transform=ax_streak.transAxes,
            fontsize=9,
            fontweight="bold",
            color="#59B37B",
        )

        # Medication adherence
        ax_med = fig.add_subplot(grid[1, 2])
        ax_med.axis("off")
        _add_rounded_panel(ax_med, "#FFE9A6")
        ax_med.text(
            0.5,
            0.95,
            "Medication\nAdherence",
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="center",
            color="#E39A00",
        )
        # Adherence percentage
        if adh_pct is not None:
            ax_med.text(
                0.5,
                0.1,
                f"{adh_pct:.0f}%",
                fontsize=34,
                fontweight="bold",
                va="center",
                ha="center",
                color="#E39A00",
            )
        else:
            ax_med.text(
                0.5,
                0.1,
                "No medication data",
                fontsize=14,
                fontstyle="italic",
                va="center",
                ha="center",
                color="#E39A00",
            )
        # Add meds icon
        try:
            icon_path = processed_dir.parent.parent / "img" / "meds.png"
            if icon_path.exists():
                arr = plt.imread(icon_path)
                imagebox = OffsetImage(arr, zoom=0.17)
                ab = AnnotationBbox(
                    imagebox,
                    (0.50, 0.48),
                    frameon=False,
                    xycoords="axes fraction",
                    zorder=2,
                )
                ax_med.add_artist(ab)
        except Exception:
            pass

        # App Usage
        ax_usage = fig.add_subplot(grid[1, 3])
        ax_usage.axis("off")
        _add_rounded_panel(ax_usage, "#E6D6FF")
        ax_usage.text(
            0.5,
            0.95,
            "App\nUsage",
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="center",
            color="#6B2FA0",
        )
        # Descriptor and value
        ax_usage.text(
            0.75,
            0.1,
            "days",
            transform=ax_usage.transAxes,
            fontsize=12,
            va="top",
            ha="left",
            color="#6B2FA0",
        )
        # Display usage as two-digit values
        padded_total = "00"
        try:
            if usage_total is not None and not pd.isna(usage_total):
                padded_total = f"{int(usage_total):02d}"
        except Exception:
            pass
        try:
            padded_lookback = f"{int(config.METRICS_LOOKBACK_DAYS):02d}"
        except Exception:
            padded_lookback = str(config.METRICS_LOOKBACK_DAYS)

        ax_usage.text(
            0.07,
            0.000001,
            f"{padded_total}/{padded_lookback}",
            transform=ax_usage.transAxes,
            fontsize=30,
            fontweight="bold",
            va="bottom",
            ha="left",
            color="#6B2FA0",
        )
        # Add phone icon
        try:
            icon_path = processed_dir.parent.parent / "img" / "phone.png"
            if icon_path.exists():
                arr = plt.imread(icon_path)
                imagebox = OffsetImage(arr, zoom=0.035)
                ab = AnnotationBbox(
                    imagebox,
                    (0.50, 0.47),
                    frameon=False,
                    xycoords="axes fraction",
                    zorder=2,
                )
                ax_usage.add_artist(ab)
        except Exception:
            pass

        # Bottom placeholders
        ax_current = fig.add_subplot(grid[2, 0])
        ax_current.axis("off")
        _add_rounded_panel(ax_current, "#FFD7EF")
        ax_current.text(
            0.5,
            0.95,
            "Current\nProspective\nSurvey",
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="center",
            color="#A1366E",
        )
        status_text = "Complete" if prospective_completed else "Incomplete"
        ax_current.text(
            0.5,
            0.01,
            status_text,
            transform=ax_current.transAxes,
            fontsize=22,
            fontweight="bold",
            va="bottom",
            ha="center",
            color="#A1366E",
        )
        try:
            icon_file = "survey_1.png" if prospective_completed else "survey_0.png"
            icon_path = processed_dir.parent.parent / "img" / icon_file
            if icon_path.exists():
                arr = plt.imread(icon_path)
                imagebox = OffsetImage(arr, zoom=0.07)
                ab = AnnotationBbox(
                    imagebox,
                    (0.5, 0.41),
                    frameon=False,
                    xycoords="axes fraction",
                    zorder=2,
                )
                ax_current.add_artist(ab)
        except Exception:
            pass

        ax_future_primary = fig.add_subplot(grid[2, 1])
        ax_future_primary.axis("off")
        _add_rounded_panel(ax_future_primary, "#D3FFF3")
        ax_future_primary.text(
            0.5,
            0.95,
            "Upcoming\nProspective\nSurvey",
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="center",
            color="#1F8A70",
        )
        ax_future_primary.text(
            0.5,
            0.01,
            upcoming_survey_label,
            transform=ax_future_primary.transAxes,
            fontsize=22,
            fontweight="bold",
            va="bottom",
            ha="center",
            color="#1F8A70",
        )
        try:
            icon_path = processed_dir.parent.parent / "img" / "survey_2.png"
            if icon_path.exists():
                arr = plt.imread(icon_path)
                imagebox = OffsetImage(arr, zoom=0.07)
                ab = AnnotationBbox(
                    imagebox,
                    (0.5, 0.41),
                    frameon=False,
                    xycoords="axes fraction",
                    zorder=2,
                )
                ax_future_primary.add_artist(ab)
        except Exception:
            pass

        ax_final_secondary = fig.add_subplot(grid[2, 2])
        ax_final_secondary.axis("off")
        _add_rounded_panel(ax_final_secondary, "#EDE0C8")
        ax_final_secondary.text(
            0.5,
            0.95,
            "Final\nProspective\nSurvey",
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="center",
            color="#8B6F3F",
        )
        ax_final_secondary.text(
            0.5,
            0.01,
            final_survey_label,
            transform=ax_final_secondary.transAxes,
            fontsize=22,
            fontweight="bold",
            va="bottom",
            ha="center",
            color="#8B6F3F",
        )
        try:
            icon_path = processed_dir.parent.parent / "img" / "survey_2.png"
            if icon_path.exists():
                arr = plt.imread(icon_path)
                imagebox = OffsetImage(arr, zoom=0.07)
                ab = AnnotationBbox(
                    imagebox,
                    (0.5, 0.41),
                    frameon=False,
                    xycoords="axes fraction",
                    zorder=2,
                )
                ax_final_secondary.add_artist(ab)
        except Exception:
            pass

        ax_whatsapp = fig.add_subplot(grid[2, 3])
        ax_whatsapp.axis("off")
        _add_rounded_panel(ax_whatsapp, "#DDE4FF")
        ax_whatsapp.text(
            0.5,
            0.95,
            "WhatsApp\nGroup",
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="center",
            color="#2A3F8F",
        )
        ax_whatsapp.text(
            0.5,
            0.01,
            whatsapp_status_label,
            transform=ax_whatsapp.transAxes,
            fontsize=whatsapp_status_fontsize,
            fontweight="bold",
            va="bottom",
            ha="center",
            color="#2A3F8F",
        )
        try:
            icon_name = "whatsapp_1.png" if whatsapp_member else "whatsapp_0.png"
            icon_path = processed_dir.parent.parent / "img" / icon_name
            if icon_path.exists():
                arr = plt.imread(icon_path)
                imagebox = OffsetImage(arr, zoom=0.22)
                ab = AnnotationBbox(
                    imagebox,
                    (0.5, 0.43),
                    frameon=False,
                    xycoords="axes fraction",
                    zorder=2,
                )
                ax_whatsapp.add_artist(ab)
        except Exception:
            pass

        fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.06)
        fig.text(
            0.99,
            0.005,
            f"Created on {date.today().strftime('%d-%m-%Y')}.",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#2F2F2F",
        )

        out_path = out_dir / f"{pid}_summary.png"
        fig.savefig(
            out_path,
            dpi=300,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.close(fig)

        written.append(out_path)

    return written
