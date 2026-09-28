"""Generate the synthetic demo dataset shipped in ``sample_data/``.

**The output is fully synthetic** — it is produced by numpy's random number
generator with a fixed seed and describes fictional students. It exists only to
demonstrate DATA AUTOPSY's detectors, so it is deliberately built to contain:

======================  ====================================================
Feature in the data     What it demonstrates
======================  ====================================================
missing values          columns at ~2%, ~6%, ~15% and ~25% missingness
duplicate rows          ~0.5% exact duplicates
numeric-as-text         ``family_income_ksh`` stored as "45,000" strings
mixed-format text       ``previous_school_score`` mixing numbers and "not available"
datetime-as-text        ``enrollment_date`` holding ISO date strings
constant column         ``extract_source`` never changes
near-constant column    ``school_type`` is "Public" for >99% of rows
outliers                extreme but plausible ages, a negative exam score
sentinel-like values    ``exam_score`` of -1 and 0, ``sleep_hours`` of 30
correlation             study/attendance/prior GPA drive exam score
duplicate feature       ``prior_gpa_percent`` is a rescaled copy of ``prior_gpa``
potential leakage       ``is_at_risk`` is derived from attendance and score
imbalance               ``internet_access`` is "Yes" for ~97% of rows
======================  ====================================================

Run with::

    python sample_data/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20240927
N_ROWS = 1_500
OUTPUT = Path(__file__).resolve().parent / "students.csv"

REGIONS = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika", "Other"]
PARENTAL_EDUCATION = [
    "No formal education", "Primary", "Secondary", "Certificate", "Diploma",
    "Bachelor", "Postgraduate",
]
GENDERS = ["Female", "Male", "Prefer not to say"]


def _grades(scores: np.ndarray) -> list[str]:
    """Convert exam scores into letter grades (a deterministic leak by design)."""
    bins = [0, 40, 50, 60, 70, 85, 101]
    labels = ["E", "D", "C", "B", "A", "A+"]
    return pd.cut(scores, bins=bins, labels=labels, right=False).astype(str).tolist()


def build_frame() -> pd.DataFrame:
    """Create the synthetic student dataset."""
    rng = np.random.default_rng(SEED)
    n = N_ROWS

    # --- latent "ability" drives both study behaviour and outcomes ----------
    ability = rng.normal(0, 0.8, n)
    study_hours = np.clip(6 + 3.2 * ability + rng.normal(0, 3.0, n), 0, 35)
    attendance_rate = np.clip(82 + 7.5 * ability + rng.normal(0, 9.0, n), 20, 100)
    prior_gpa = np.clip(2.7 + 0.42 * ability + rng.normal(0, 0.35, n), 0.8, 4.0)
    sleep_hours = np.clip(7.0 + rng.normal(0, 1.15, n), 2.5, 11.0)
    assignments_completed = np.clip(
        np.round(18 + 5 * ability + rng.normal(0, 4.0, n)), 0, 30
    ).astype(int)
    age = np.clip(np.round(rng.normal(19.5, 1.7, n)), 16, 26).astype(int)

    # Each contributor is centred on its own mean, so only its *variation* moves the
    # score: the marks stay centred near 62 with a standard deviation of about 11.5,
    # and each feature's influence is expressed directly as its coefficient.
    exam_score = (
        62.0
        + 1.05 * (study_hours - study_hours.mean())
        + 0.40 * (attendance_rate - attendance_rate.mean())
        + 8.50 * (prior_gpa - prior_gpa.mean())
        + 0.48 * (assignments_completed - assignments_completed.mean())
        + 0.40 * (sleep_hours - sleep_hours.mean())
        + rng.normal(0, 6.0, n)
    )
    exam_score = np.clip(exam_score, 5, 100)

    frame = pd.DataFrame(
        {
            "student_id": [f"STU-{i:05d}" for i in range(1, n + 1)],
            "age": age,
            "gender": rng.choice(GENDERS, n, p=[0.51, 0.46, 0.03]),
            "region": rng.choice(
                REGIONS, n, p=[0.34, 0.18, 0.14, 0.12, 0.09, 0.07, 0.06]
            ),
            "study_hours_per_week": np.round(study_hours, 1),
            "attendance_rate": np.round(attendance_rate, 1),
            "prior_gpa": np.round(prior_gpa, 2),
            "assignments_completed": assignments_completed,
            "sleep_hours": np.round(sleep_hours, 1),
            "exam_score": np.round(exam_score, 1),
            "final_grade": _grades(exam_score),
            "parental_education": rng.choice(
                PARENTAL_EDUCATION, n, p=[0.05, 0.14, 0.26, 0.13, 0.18, 0.19, 0.05]
            ),
            "internet_access": rng.choice(["Yes", "No"], n, p=[0.972, 0.028]),
            "school_type": rng.choice(["Public", "Private"], n, p=[0.994, 0.006]),
            "family_income_ksh": [
                f"{int(value):,}" for value in np.clip(rng.normal(48_000, 21_000, n), 5_000, 400_000)
            ],
            "previous_school_score": [
                "not available" if rng.random() < 0.18 else str(int(value))
                for value in np.clip(rng.normal(66 + 4.5 * ability, 9.0, n), 20, 100)
            ],
            "enrollment_date": [
                f"2024-{month:02d}-{day:02d}"
                for month, day in zip(
                    rng.integers(1, 13, n), rng.integers(1, 29, n)
                )
            ],
            "extract_source": "synthetic_demo",
        }
    )

    # --- deliberate realistic imperfections ---------------------------------
    # 1. Missing values, with sleep_hours missing more often for low scorers
    #    (a mildly non-random mechanism — worth investigating, by design).
    for column, rate in (
        ("attendance_rate", 0.03),
        ("prior_gpa", 0.06),
        ("sleep_hours", 0.12),
        ("parental_education", 0.15),
        ("family_income_ksh", 0.25),
    ):
        mask = rng.random(n) < rate
        if column == "sleep_hours":
            mask |= (frame["exam_score"] < 45) & (rng.random(n) < 0.20)
        frame.loc[mask, column] = np.nan

    # 2. Extreme but plausible observations (adult learners returning to study).
    adult_learners = rng.choice(n, 7, replace=False)
    frame.loc[adult_learners, "age"] = rng.integers(31, 48, adult_learners.size)
    frame.loc[adult_learners, "study_hours_per_week"] = rng.uniform(22, 34, adult_learners.size).round(1)

    # 3. Data-entry problems a real extract would contain.
    frame.loc[rng.choice(n, 4, replace=False), "exam_score"] = -1.0     # placeholder code
    frame.loc[rng.choice(n, 6, replace=False), "exam_score"] = 0.0      # did not sit the exam
    frame.loc[rng.choice(n, 3, replace=False), "sleep_hours"] = 30.0    # impossible value
    frame.loc[rng.choice(n, 5, replace=False), "attendance_rate"] = 0.0

    # 4. A rescaled duplicate of prior_gpa (a common "duplicate feature").
    frame["prior_gpa_percent"] = (frame["prior_gpa"] / 4.0 * 100).round(2)

    # 5. The prediction target. It is derived *directly* from exam_score, which means
    #    exam_score leaks the outcome — exactly the situation the leakage screen must
    #    catch, and the reason a single feature separates the target perfectly.
    frame["is_at_risk"] = (frame["exam_score"] < 55).astype(int)

    # 6. Exact duplicate rows (export artefact).
    duplicate_rows = frame.sample(8, random_state=SEED)
    frame = pd.concat([frame, duplicate_rows], ignore_index=True)

    # 7. Column order: identifiers first, target last (the common ML layout).
    ordered = [
        "student_id", "age", "gender", "region", "school_type", "study_hours_per_week",
        "attendance_rate", "prior_gpa", "prior_gpa_percent", "assignments_completed",
        "sleep_hours", "exam_score", "family_income_ksh", "previous_school_score",
        "parental_education", "internet_access", "enrollment_date", "final_grade",
        "extract_source", "is_at_risk",
    ]
    return frame[ordered]


def main() -> None:
    """Write the synthetic dataset to ``sample_data/students.csv``."""
    frame = build_frame()
    frame.to_csv(OUTPUT, index=False)
    print(
        f"Wrote {OUTPUT} — {len(frame):,} rows x {frame.shape[1]} columns "
        "(fully synthetic; see the module docstring)."
    )


if __name__ == "__main__":
    main()
