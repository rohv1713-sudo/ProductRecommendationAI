"""
Carbon Fiber Recommendation Engine v5

Carbon-fiber-only recommender.

Required database:
    Carbon_Fiber_Master_v6_Top5(1).csv

Prepreg and resin databases are NOT loaded or required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ============================================================
# DATABASE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Preferred filename
FIBER_FILE = BASE_DIR / "Carbon_Fiber_Master_v6_Top5(1).csv"

# Fallback filenames
if not FIBER_FILE.exists():
    for name in [
        "Carbon_Fiber_Master_v6_Top5.csv",
        "Carbon_Fiber_Master.csv",
    ]:
        candidate = BASE_DIR / name
        if candidate.exists():
            FIBER_FILE = candidate
            break

# ============================================================
# APPLICATIONS
# ============================================================

APPLICATIONS = {
    "1": {
        "name": "Aerospace",
        "importance": {"strength": 9, "modulus": 10, "density": 10, "elongation": 6},
        "use_terms": ["aerospace"],
    },
    "2": {
        "name": "Automotive / Motorsport",
        "importance": {"strength": 8, "modulus": 8, "density": 9, "elongation": 7},
        "use_terms": ["motorsport", "automotive", "industrial"],
    },
    "3": {
        "name": "Pressure Vessel / Energy",
        "importance": {"strength": 10, "modulus": 8, "density": 8, "elongation": 8},
        "use_terms": ["pressure vessel", "energy", "industrial"],
    },
    "4": {
        "name": "Sporting Goods",
        "importance": {"strength": 7, "modulus": 9, "density": 9, "elongation": 7},
        "use_terms": ["sporting goods"],
    },
    "5": {
        "name": "Industrial",
        "importance": {"strength": 8, "modulus": 7, "density": 7, "elongation": 6},
        "use_terms": ["industrial"],
    },
    "6": {
        "name": "Custom",
        "importance": None,
        "use_terms": [],
    },
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    return clean_text(value).lower()


def number(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))

    return float(match.group()) if match else np.nan


def parse_tows(value: Any) -> set[int]:
    if pd.isna(value):
        return set()

    return {
        int(x)
        for x in re.findall(r"(\d+)\s*K", str(value).upper())
    }


def ratio_score(
    value: float,
    target: float,
    higher_is_better: bool
) -> float:

    if not np.isfinite(value) or not np.isfinite(target) or target <= 0:
        return np.nan

    ratio = value / target

    if higher_is_better:
        return float(
            np.clip(ratio, 0.0, 1.5) / 1.5 * 100.0
        )

    if value <= target:
        return 100.0

    return float(
        np.clip(target / value, 0.0, 1.0) * 100.0
    )


def weighted_mean(
    values: Dict[str, float],
    weights: Dict[str, float]
) -> float:

    usable = {
        k: v
        for k, v in values.items()
        if np.isfinite(v)
    }

    if not usable:
        return np.nan

    total_w = sum(weights[k] for k in usable)

    if total_w <= 0:
        return np.nan

    return sum(
        usable[k] * weights[k]
        for k in usable
    ) / total_w


# ============================================================
# CARBON-FIBER DATABASE LOADING
# ============================================================

def normalize_fiber_db(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    required = [
        "fiber_id",
        "manufacturer",
        "product",
        "tensile_strength_mpa",
        "tensile_modulus_gpa",
        "ultimate_elongation_percent",
        "fiber_density_g_cm3",
    ]

    missing = [
        c for c in required
        if c not in out.columns
    ]

    if missing:
        raise ValueError(
            "Carbon-fiber database is missing columns: "
            + str(missing)
        )

    numeric = [
        "tensile_strength_mpa",
        "tensile_modulus_gpa",
        "ultimate_elongation_percent",
        "fiber_density_g_cm3",
        "filament_diameter_um",
        "carbon_content_percent",
    ]

    for col in numeric:
        if col in out.columns:
            out[col] = out[col].apply(number)

    return out


def load_data() -> pd.DataFrame:
    """
    Load ONLY the carbon-fiber database.

    No prepreg or resin files are referenced here.
    """

    if not FIBER_FILE.exists():

        raise FileNotFoundError(
            "\nCarbon-fiber database was not found.\n\n"
            f"Expected file:\n{FIBER_FILE}\n\n"
            "Make sure the CSV is in the same folder as this script."
        )

    fibers = pd.read_csv(FIBER_FILE)

    return normalize_fiber_db(fibers)


# ============================================================
# FILTERING
# ============================================================

@dataclass
class Requirements:
    strength_mpa: float
    modulus_gpa: float
    max_density_g_cm3: float
    elongation_percent: float

    hard_strength: bool
    hard_modulus: bool
    hard_density: bool
    hard_elongation: bool

    process: Optional[str]
    matrix: Optional[str]
    require_certification: bool


def numerical_filter(
    df: pd.DataFrame,
    req: Requirements
) -> pd.DataFrame:

    mask = pd.Series(True, index=df.index)

    constraints = [
        (
            "tensile_strength_mpa",
            req.strength_mpa,
            req.hard_strength,
            True,
        ),
        (
            "tensile_modulus_gpa",
            req.modulus_gpa,
            req.hard_modulus,
            True,
        ),
        (
            "ultimate_elongation_percent",
            req.elongation_percent,
            req.hard_elongation,
            True,
        ),
        (
            "fiber_density_g_cm3",
            req.max_density_g_cm3,
            req.hard_density,
            False,
        ),
    ]

    for col, threshold, hard, higher_is_better in constraints:

        if not hard:
            continue

        valid = df[col].notna()

        if higher_is_better:
            mask &= valid & (df[col] >= threshold)
        else:
            mask &= valid & (df[col] <= threshold)

    return df.loc[mask].copy()


# ============================================================
# FIT / SCORING
# ============================================================

def application_fit(
    row: pd.Series,
    app: dict
) -> float:

    terms = app["use_terms"]

    if not terms:
        return 100.0

    text = " ".join(
        norm(row.get(col, ""))
        for col in [
            "recommended_uses",
            "market_or_grade",
            "fiber_class",
        ]
    )

    hits = sum(term in text for term in terms)

    if hits >= 2:
        return 100.0

    if hits == 1:
        return 75.0

    return 50.0


def grade_fit(
    row: pd.Series,
    application_name: str
) -> float:

    text = " ".join(
        norm(row.get(col, ""))
        for col in [
            "market_or_grade",
            "fiber_class",
            "recommended_uses",
        ]
    )

    if application_name == "Aerospace":
        return 100.0 if "aerospace" in text else 50.0

    if application_name in {
        "Automotive / Motorsport",
        "Industrial",
        "Pressure Vessel / Energy",
    }:
        return 100.0 if "industrial" in text else 70.0

    return 100.0


def process_fit(
    row: pd.Series,
    process: Optional[str]
) -> float:

    if not process:
        return 100.0

    process_text = " ".join(
        norm(row.get(col, ""))
        for col in [
            "recommended_uses",
            "sizing_compatibility",
        ]
    )

    tows = parse_tows(
        row.get("sizing_options", "")
    )

    if process in {"weaving", "braiding"}:
        return (
            100.0
            if tows & {3, 6, 12, 24}
            else 55.0
        )

    if process == "thin prepreg tape":
        return (
            100.0
            if tows & {3, 6}
            else 70.0
        )

    if process in {
        "prepreg tape",
        "filament winding",
        "heavy stitched/woven",
    }:
        return (
            100.0
            if tows & {12, 24}
            else 65.0
        )

    if process == "pultrusion":
        return (
            100.0
            if "pultrusion" in process_text
            else 60.0
        )

    return 70.0


def fiber_matrix_fit(
    row: pd.Series,
    matrix: Optional[str]
) -> float:

    if not matrix:
        return 100.0

    compatibility = norm(
        row.get("sizing_compatibility", "")
    )

    sizing = norm(
        row.get("sizing_options", "")
    )

    if matrix == "thermoplastic":
        return (
            100.0
            if (
                "unsized" in sizing
                or "thermoplastic" in compatibility
            )
            else 45.0
        )

    if matrix in compatibility:
        return 100.0

    if matrix == "epoxy" and any(
        s in sizing
        for s in ["gp", "gs", " g", "g"]
    ):
        return 95.0

    return 60.0


def certification_fit(
    row: pd.Series,
    required: bool
) -> float:

    if not required:
        return 100.0

    cert = norm(
        row.get("certification", "")
    )

    return (
        100.0
        if cert and cert not in {"not stated", "nan"}
        else 0.0
    )


def engineering_score(
    row: pd.Series,
    req: Requirements,
    app: dict
) -> float:

    values = {
        "strength": ratio_score(
            row["tensile_strength_mpa"],
            req.strength_mpa,
            True,
        ),

        "modulus": ratio_score(
            row["tensile_modulus_gpa"],
            req.modulus_gpa,
            True,
        ),

        "elongation": ratio_score(
            row["ultimate_elongation_percent"],
            req.elongation_percent,
            True,
        ),

        "density": ratio_score(
            row["fiber_density_g_cm3"],
            req.max_density_g_cm3,
            False,
        ),
    }

    importance = app["importance"]

    weights = {
        key: val / sum(importance.values())
        for key, val in importance.items()
    }

    return weighted_mean(values, weights)


def build_ranked_results(
    df: pd.DataFrame,
    req: Requirements,
    app: dict
) -> pd.DataFrame:

    candidates = numerical_filter(df, req)

    if req.require_certification:

        cert_mask = (
            candidates["certification"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

        candidates = candidates.loc[
            cert_mask
        ].copy()

    if candidates.empty:
        return pd.DataFrame()

    records = []

    for _, row in candidates.iterrows():

        eng = engineering_score(
            row,
            req,
            app
        )

        app_score = application_fit(
            row,
            app
        )

        grade_score_value = grade_fit(
            row,
            app["name"]
        )

        proc = process_fit(
            row,
            req.process
        )

        matrix = fiber_matrix_fit(
            row,
            req.matrix
        )

        cert = certification_fit(
            row,
            req.require_certification
        )

        final = (
            0.70 * eng
            + 0.08 * app_score
            + 0.06 * grade_score_value
            + 0.08 * proc
            + 0.06 * matrix
            + 0.02 * cert
        )

        records.append({
            "Fiber_ID": row["fiber_id"],
            "Manufacturer": row["manufacturer"],
            "Product": row["product"],
            "Strength_MPa": row["tensile_strength_mpa"],
            "Modulus_GPa": row["tensile_modulus_gpa"],
            "Density_g_cm3": row["fiber_density_g_cm3"],
            "Elongation_percent": row[
                "ultimate_elongation_percent"
            ],
            "Tow_or_Sizing": row.get(
                "sizing_options", ""
            ),
            "Fiber_Class": row.get(
                "fiber_class", ""
            ),
            "Certification": row.get(
                "certification", ""
            ),
            "Engineering_Score": round(
                eng, 2
            ),
            "Application_Fit": round(
                app_score, 2
            ),
            "Grade_Fit": round(
                grade_score_value, 2
            ),
            "Process_Fit": round(
                proc, 2
            ),
            "Matrix_Fit": round(
                matrix, 2
            ),
            "Certification_Fit": round(
                cert, 2
            ),
            "Final_Score": round(
                final, 2
            ),
            "Source_File": row.get(
                "source_file", ""
            ),
            "Source_URL": row.get(
                "source_url", ""
            ),
            "Data_Status": row.get(
                "fiber_data_status", ""
            ),
        })

    result = (
        pd.DataFrame(records)
        .sort_values(
            "Final_Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    if not result.empty:

        result["Rank"] = np.arange(
            1,
            len(result) + 1
        )

        result["Relative_Score_percent"] = (
            100.0
            * result["Final_Score"]
            / result["Final_Score"].max()
        ).round(1)

    return result


# ============================================================
# OUTPUT
# ============================================================

def print_results(
    result: pd.DataFrame,
    req: Requirements,
    app: dict
) -> None:

    print("\n" + "=" * 70)
    print("CARBON FIBER RECOMMENDATIONS")
    print("=" * 70)

    print(
        f"\nApplication: {app['name']}\n"
        f"Process: {req.process or 'None'}\n"
        f"Matrix: {req.matrix or 'None'}"
    )

    if result.empty:

        print(
            "\nNO CARBON-FIBER CANDIDATE SATISFIES "
            "THE HARD REQUIREMENTS."
        )

        return

    print(
        f"\nShowing top {min(5, len(result))} "
        f"of {len(result)} candidates:"
    )

    for _, r in result.head(5).iterrows():

        print(
            f"\n#{int(r['Rank'])} "
            f"{r['Manufacturer']} "
            f"{r['Product']} "
            f"— {r['Relative_Score_percent']:.1f}%"
        )

        print(
            f"  Strength: {r['Strength_MPa']:.0f} MPa | "
            f"Modulus: {r['Modulus_GPa']:.0f} GPa | "
            f"Density: {r['Density_g_cm3']:.2f} g/cm³ | "
            f"Elongation: {r['Elongation_percent']:.1f}%"
        )

        print(
            f"  Engineering: {r['Engineering_Score']:.1f} | "
            f"Application: {r['Application_Fit']:.1f} | "
            f"Process: {r['Process_Fit']:.1f} | "
            f"Matrix: {r['Matrix_Fit']:.1f}"
        )

        print(
            f"  Fiber class: {r['Fiber_Class']}"
        )

        print(
            f"  Sizing: {r['Tow_or_Sizing']}"
        )

        print(
            f"  Certification: "
            f"{r['Certification'] or 'Not stated'}"
        )


# ============================================================
# INPUT HELPERS
# ============================================================

def ask_float(
    prompt: str,
    minimum: float = 0.0
) -> float:

    while True:

        try:
            value = float(
                input(prompt)
            )

            if value < minimum:
                raise ValueError

            return value

        except ValueError:

            print(
                f"Enter a number >= {minimum}."
            )


def ask_yes_no(prompt: str) -> bool:

    while True:

        answer = (
            input(prompt)
            .strip()
            .lower()
        )

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Enter yes or no.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print(" CARBON FIBER RECOMMENDATION ENGINE v5")
    print("=" * 70)

    print(
        f"\nDatabase: {FIBER_FILE.name}"
    )

    fibers = load_data()

    print(
        f"Loaded {len(fibers)} carbon-fiber grades."
    )

    print("\nAPPLICATION")

    for key, value in APPLICATIONS.items():
        print(
            f"{key}. {value['name']}"
        )

    choice = input(
        "Choose application: "
    ).strip()

    while choice not in APPLICATIONS:

        choice = input(
            "Choose a valid option: "
        ).strip()

    app = APPLICATIONS[choice].copy()

    if choice == "6":

        print(
            "\nCUSTOM IMPORTANCE (1-10)"
        )

        app["importance"] = {
            "strength": ask_float(
                "Strength importance: ",
                1
            ),

            "modulus": ask_float(
                "Modulus importance: ",
                1
            ),

            "density": ask_float(
                "Density importance: ",
                1
            ),

            "elongation": ask_float(
                "Elongation importance: ",
                1
            ),
        }

    print(
        "\nENGINEERING REQUIREMENTS"
    )

    req = Requirements(

        strength_mpa=ask_float(
            "Minimum tensile strength (MPa): "
        ),

        modulus_gpa=ask_float(
            "Minimum tensile modulus (GPa): "
        ),

        max_density_g_cm3=ask_float(
            "Maximum density (g/cm³): "
        ),

        elongation_percent=ask_float(
            "Minimum elongation (%): "
        ),

        hard_strength=ask_yes_no(
            "Strength is HARD? "
        ),

        hard_modulus=ask_yes_no(
            "Modulus is HARD? "
        ),

        hard_density=ask_yes_no(
            "Density is HARD? "
        ),

        hard_elongation=ask_yes_no(
            "Elongation is HARD? "
        ),

        process=None,
        matrix=None,
        require_certification=False,
    )

    # Keep the original process/matrix inputs as fiber-level
    # compatibility preferences. They do NOT load prepreg/resin data.
    process_options = {
        "1": None,
        "2": "weaving",
        "3": "thin prepreg tape",
        "4": "prepreg tape",
        "5": "filament winding",
        "6": "heavy stitched/woven",
        "7": "pultrusion",
        "8": "braiding",
    }

    matrix_options = {
        "1": None,
        "2": "epoxy",
        "3": "thermoplastic",
        "4": "polycarbonate",
        "5": "phenolic",
        "6": "vinyl ester",
        "7": "cyanate ester",
        "8": "BMI",
        "9": "polyimide",
    }

    print("\nPROCESS")

    for key, name in process_options.items():
        print(
            f"{key}. "
            f"{name or 'No process preference'}"
        )

    req.process = process_options.get(
        input(
            "Choose process: "
        ).strip()
    )

    print("\nMATRIX")

    for key, name in matrix_options.items():
        print(
            f"{key}. "
            f"{name or 'No matrix preference'}"
        )

    req.matrix = matrix_options.get(
        input(
            "Choose matrix: "
        ).strip()
    )

    req.require_certification = ask_yes_no(
        "\nRequire an explicitly stated certification? "
    )

    result = build_ranked_results(
        fibers,
        req,
        app
    )

    print_results(
        result,
        req,
        app
    )

    if not result.empty:

        export = ask_yes_no(
            "\nExport complete ranking to CSV? "
        )

        if export:

            out = (
                BASE_DIR
                / "carbon_fiber_recommendation_results.csv"
            )

            result.to_csv(
                out,
                index=False
            )

            print(
                f"Saved: {out}"
            )


if __name__ == "__main__":
    main()
