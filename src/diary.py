"""Helpers for extracting SCN8A weekly diary responses."""

from __future__ import annotations

import re
from typing import Callable, Iterable

import pandas as pd

_FORM_MATCH_KEYWORD = "scn8a diary completion"
_SLEEP_QUESTION_KEYWORDS: tuple[str, ...] = ("sleep quality",)
_SLEEP_QUESTION_CODE_TOKENS: tuple[str, ...] = (
    "q4_4_on_a_scale_of_1_10_how_was_your_child_s_sleep_quality_this_week",
)
_SLEEP_CODE_SUBSTRINGS: tuple[str, ...] = ("sleep_quality",)

_MOOD_QUESTION_KEYWORDS: tuple[str, ...] = ("mood",)
_MOOD_QUESTION_CODE_TOKENS: tuple[str, ...] = (
    "q3_3_on_a_scale_of_1_10_how_was_your_child_s_mood_this_week",
)
_MOOD_CODE_SUBSTRINGS: tuple[str, ...] = ("mood",)

_MED_ADHERENCE_KEYWORDS: tuple[str, ...] = ("give your child all medications",)
_MED_ADHERENCE_CODE_TOKENS: tuple[str, ...] = (
    "q1_1_did_you_give_your_child_all_medications_every_day_this_week",
)
_MED_ADHERENCE_CODE_SUBSTRINGS: tuple[str, ...] = ("give_your_child_all_medications",)
_RESULT_COLUMNS = ["patient_id", "date", "score"]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a defensive copy of *df* that includes all *columns*."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def extract_sleep_diary_entries(
    forms_df: pd.DataFrame | None, answers_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Return normalized SCN8A diary sleep responses with columns
    (patient_id, date, score)."""
    if forms_df is None or answers_df is None:
        return pd.DataFrame(columns=_RESULT_COLUMNS)
    if forms_df.empty or answers_df.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    return _extract_diary_question(
        forms_df,
        answers_df,
        question_keywords=_SLEEP_QUESTION_KEYWORDS,
        question_code_tokens=_SLEEP_QUESTION_CODE_TOKENS,
        question_code_substrings=_SLEEP_CODE_SUBSTRINGS,
    )


def extract_mood_diary_entries(
    forms_df: pd.DataFrame | None, answers_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Return normalized SCN8A diary mood responses."""
    return _extract_diary_question(
        forms_df,
        answers_df,
        question_keywords=_MOOD_QUESTION_KEYWORDS,
        question_code_tokens=_MOOD_QUESTION_CODE_TOKENS,
        question_code_substrings=_MOOD_CODE_SUBSTRINGS,
    )


def extract_med_adherence_diary_entries(
    forms_df: pd.DataFrame | None, answers_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Return SCN8A diary medication adherence responses mapped to 0-100%."""
    return _extract_diary_question(
        forms_df,
        answers_df,
        question_keywords=_MED_ADHERENCE_KEYWORDS,
        question_code_tokens=_MED_ADHERENCE_CODE_TOKENS,
        question_code_substrings=_MED_ADHERENCE_CODE_SUBSTRINGS,
        value_parser=_parse_med_adherence_pct,
    )


def _extract_diary_question(
    forms_df: pd.DataFrame | None,
    answers_df: pd.DataFrame | None,
    *,
    question_keywords: Iterable[str] = (),
    question_code_tokens: Iterable[str] = (),
    question_code_substrings: Iterable[str] = (),
    value_parser: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """Shared extractor for diary questions."""
    if forms_df is None or answers_df is None:
        return pd.DataFrame(columns=_RESULT_COLUMNS)
    if forms_df.empty or answers_df.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    forms = _ensure_columns(
        forms_df, ("form_id", "patient_id", "date", "form_name", "label")
    )
    answers = _ensure_columns(
        answers_df, ("form_id", "question_code", "question_text", "answer")
    )
    answers = answers.drop(columns=["patient_id"], errors="ignore")

    forms = forms.dropna(subset=["form_id", "patient_id"])
    if forms.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    # Identify SCN8A Diary completion forms
    match_text = (
        forms["form_name"].fillna("").astype(str)
        + "||"
        + forms["label"].fillna("").astype(str)
    ).str.lower()
    diary_forms = forms[match_text.str.contains(_FORM_MATCH_KEYWORD, na=False)]
    if diary_forms.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    diary_forms = diary_forms.copy()
    diary_forms["form_id"] = diary_forms["form_id"].astype(str)
    diary_forms["patient_id"] = diary_forms["patient_id"].astype(str)
    diary_forms["_date"] = pd.to_datetime(
        diary_forms["date"], utc=True, errors="coerce"
    )
    diary_forms = diary_forms[diary_forms["_date"].notna()]
    if diary_forms.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    # Filter to the requested question
    answers = answers.dropna(subset=["form_id"])
    if answers.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    answers = answers.copy()
    answers["form_id"] = answers["form_id"].astype(str)
    answers["_q_text"] = answers["question_text"].fillna("").astype(str).str.lower()
    answers["_q_code"] = answers["question_code"].fillna("").astype(str).str.lower()

    mask = pd.Series(False, index=answers.index)
    keyword_pattern = _compile_pattern(question_keywords)
    code_pattern = _compile_pattern(question_code_substrings)
    tokens_lower = {tok.lower() for tok in question_code_tokens if tok}

    if keyword_pattern:
        mask |= answers["_q_text"].str.contains(keyword_pattern, na=False)
    if code_pattern:
        mask |= answers["_q_code"].str.contains(code_pattern, na=False)
    if tokens_lower:
        mask |= answers["_q_code"].isin(tokens_lower)

    filtered_answers = answers[mask]
    if filtered_answers.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    filtered_answers = filtered_answers.copy()
    parser = value_parser or _parse_numeric_scale
    scores = parser(filtered_answers)
    filtered_answers["score"] = pd.to_numeric(scores, errors="coerce")
    filtered_answers = filtered_answers[filtered_answers["score"].notna()]
    if filtered_answers.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    merged = filtered_answers.merge(
        diary_forms[["form_id", "patient_id", "_date"]],
        on="form_id",
        how="inner",
    )
    merged = merged.dropna(subset=["patient_id", "_date", "score"])
    if merged.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    merged = merged.copy()
    merged["patient_id"] = merged["patient_id"].astype(str)
    merged["date"] = merged["_date"].dt.tz_localize(None)

    result = (
        merged[["patient_id", "date", "score"]]
        .sort_values(["patient_id", "date"])
        .reset_index(drop=True)
    )
    return result


def _parse_numeric_scale(df: pd.DataFrame) -> pd.Series:
    """Parse numeric responses mapped to a 0-10 scale."""
    values = pd.to_numeric(df["answer"], errors="coerce")
    return values.astype(float).clip(0.0, 10.0)


def _parse_med_adherence_pct(df: pd.DataFrame) -> pd.Series:
    """Map yes/no medication adherence answers to a percentage."""
    tokens = (
        df["answer"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
    )
    out = pd.Series(float("nan"), index=df.index, dtype="float64")
    yes_tokens = {"yes", "y", "true", "1"}
    no_tokens = {"no", "n", "false", "0"}
    out[tokens.isin(yes_tokens)] = 100.0
    out[tokens.isin(no_tokens)] = 0.0
    return out


def _compile_pattern(tokens: Iterable[str]) -> str:
    """Return a regex OR pattern from *tokens*."""
    cleaned = [re.escape(tok.strip().lower()) for tok in tokens if tok]
    cleaned = [tok for tok in cleaned if tok]
    return "|".join(cleaned)
