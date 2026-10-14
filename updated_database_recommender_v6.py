"""
Composite Material Recommendation Engine v6

Refactor of the earlier Hexcel-only recommender.

Key changes:
- Uses the current combined carbon-fiber master database.
- Uses the original database field names and helper functions directly.
- Keeps HARD constraints separate from PREFERRED scoring.
- Uses bounded 0-100 component scores so final ranking is interpretable.
- Keeps manufacturer/source traceability in the result.
- Adds optional prepreg/resin process filtering without pretending
  generic compatibility is a manufacturer qualification.
- Leaves room for future laminate/process recommendation layers.

Database files expected beside this script (or set paths below):
    Carbon_Fiber_Master_v6_Top5(1).csv
    Prepreg_Master_Database_v2_Expanded.csv
    Resin_Master_Database_v10.csv

This engine recommends candidate MATERIAL SYSTEM INPUTS.
It does not replace engineering qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

FIBER_FILE = BASE_DIR / "Carbon_Fiber_Master_v6_Top5(1).csv"
PREPREG_FILE = BASE_DIR / "Prepreg_Master_Database_v2_Expanded.csv"
RESIN_FILE = BASE_DIR / "Resin_Master_Database_v10.csv"
COMPATIBILITY_FILE = BASE_DIR / "Compatibility_Master_v1.csv"

if not FIBER_FILE.exists():
    uploaded_fiber_file = Path("/mnt/data/Carbon_Fiber_Master_v6_Top5(1).csv")
    if uploaded_fiber_file.exists():
        FIBER_FILE = uploaded_fiber_file



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


PROCESS_OPTIONS = {
    "1": None,
    "2": "weaving",
    "3": "thin prepreg tape",
    "4": "prepreg tape",
    "5": "filament winding",
    "6": "heavy stitched/woven",
    "7": "pultrusion",
    "8": "braiding",
}

MATRIX_OPTIONS = {
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


def split_numbers(value: Any) -> list[float]:
    if pd.isna(value):
        return []
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]


def parse_tows(value: Any) -> set[int]:
    if pd.isna(value):
        return set()
    return {int(x) for x in re.findall(r"(\d+)\s*K", str(value).upper())}


def ratio_score(value: float, target: float, higher_is_better: bool) -> float:
    """Convert a requirement ratio into a bounded 0-100 preference score."""
    # Database values may arrive as strings even after CSV loading.
    value = number(value)
    target = number(target)

    if not np.isfinite(value) or not np.isfinite(target) or target <= 0:
        return np.nan

    ratio = value / target

    if higher_is_better:
        return float(np.clip(ratio, 0.0, 1.5) / 1.5 * 100.0)

    if value <= target:
        return 100.0

    return float(np.clip(target / value, 0.0, 1.0) * 100.0)


def weighted_mean(values: Dict[str, float], weights: Dict[str, float]) -> float:
    usable = {k: v for k, v in values.items() if np.isfinite(v)}
    if not usable:
        return np.nan

    total_w = sum(weights[k] for k in usable)
    if total_w <= 0:
        return np.nan

    return sum(usable[k] * weights[k] for k in usable) / total_w



def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all master databases plus the compatibility evidence layer."""
    required_files = [
        FIBER_FILE,
        PREPREG_FILE,
        RESIN_FILE,
        COMPATIBILITY_FILE,
    ]

    missing = [p for p in required_files if not p.exists()]

    if missing:
        msg = "\n".join(f"  - {p.name}" for p in missing)
        raise FileNotFoundError(
            "Required database file(s) were not found in the project folder:\n"
            + msg
        )

    fibers = pd.read_csv(FIBER_FILE)
    prepregs = pd.read_csv(PREPREG_FILE)
    resins = pd.read_csv(RESIN_FILE)
    compatibility = pd.read_csv(COMPATIBILITY_FILE)

    # These are the fields required by the locked v6 engineering logic.
    required_fiber_columns = [
        "fiber_id",
        "manufacturer",
        "product",
        "tensile_strength_mpa",
        "tensile_modulus_gpa",
        "ultimate_elongation_percent",
        "fiber_density_g_cm3",
    ]

    missing_fiber_columns = [
        col for col in required_fiber_columns
        if col not in fibers.columns
    ]

    if missing_fiber_columns:
        raise ValueError(
            "Carbon fiber database is missing required columns:\n  - "
            + "\n  - ".join(missing_fiber_columns)
        )

    # Convert engineering values using the original number() helper.
    fiber_numeric_columns = [
        "tensile_strength_mpa",
        "tensile_modulus_gpa",
        "ultimate_elongation_percent",
        "fiber_density_g_cm3",
        "filament_diameter_um",
        "carbon_content_percent",
    ]

    prepreg_numeric_columns = [
        "fiber_areal_weight_gsm",
        "resin_content_percent",
        "nominal_cured_ply_thickness_mm",
        "nominal_fiber_volume_percent",
        "nominal_laminate_density_g_cm3",
        "cure_temperature_C",
        "Tg_C",
        "service_temperature_C",
    ]

    resin_numeric_columns = [
        "density_g_cm3",
        "viscosity_value",
        "viscosity_temperature_C",
        "cure_temperature_C",
        "Tg_C",
        "service_temperature_C",
    ]

    for col in fiber_numeric_columns:
        if col in fibers.columns:
            fibers[col] = fibers[col].apply(number)

    for col in prepreg_numeric_columns:
        if col in prepregs.columns:
            prepregs[col] = prepregs[col].apply(number)

    for col in resin_numeric_columns:
        if col in resins.columns:
            resins[col] = resins[col].apply(number)

    # Compatibility level is used for sorting/scoring and must be numeric.
    if "compatibility_level" in compatibility.columns:
        compatibility["compatibility_level"] = (
            compatibility["compatibility_level"].apply(number)
        )
    else:
        compatibility["compatibility_level"] = 0.0

    return (
        fibers,
        prepregs,
        resins,
        compatibility,
    )



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


def numerical_filter(df: pd.DataFrame, req: Requirements) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    constraints = [
        ("tensile_strength_mpa", req.strength_mpa, req.hard_strength, True),
        ("tensile_modulus_gpa", req.modulus_gpa, req.hard_modulus, True),
        ("ultimate_elongation_percent", req.elongation_percent, req.hard_elongation, True),
        ("fiber_density_g_cm3", req.max_density_g_cm3, req.hard_density, False),
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


def application_fit(row: pd.Series, app: dict) -> float:
    terms = app["use_terms"]
    if not terms:
        return 100.0

    text = " ".join(
        norm(row.get(col, ""))
        for col in ["recommended_uses", "market_or_grade", "fiber_class"]
    )

    hits = sum(term in text for term in terms)
    if hits >= 2:
        return 100.0
    if hits == 1:
        return 75.0
    return 50.0


def grade_fit(row: pd.Series, application_name: str) -> float:
    text = " ".join(
        norm(row.get(col, ""))
        for col in ["market_or_grade", "fiber_class", "recommended_uses"]
    )

    if application_name == "Aerospace":
        return 100.0 if "aerospace" in text else 50.0

    if application_name in {"Automotive / Motorsport", "Industrial", "Pressure Vessel / Energy"}:
        return 100.0 if "industrial" in text else 70.0

    return 100.0


def process_fit(row: pd.Series, process: Optional[str]) -> float:
    if not process:
        return 100.0

    process_text = " ".join(
        norm(row.get(col, ""))
        for col in ["recommended_uses", "sizing_compatibility"]
    )

    tows = parse_tows(row.get("sizing_options", ""))

    if process in {"weaving", "braiding"}:
        return 100.0 if tows & {3, 6, 12, 24} else 55.0

    if process == "thin prepreg tape":
        return 100.0 if tows & {3, 6} else 70.0

    if process in {"prepreg tape", "filament winding", "heavy stitched/woven"}:
        return 100.0 if tows & {12, 24} else 65.0

    if process == "pultrusion":
        return 100.0 if "pultrusion" in process_text else 60.0

    return 70.0


def fiber_matrix_fit(row: pd.Series, matrix: Optional[str]) -> float:
    if not matrix:
        return 100.0

    compatibility = norm(row.get("sizing_compatibility", ""))
    sizing = norm(row.get("sizing_options", ""))

    if matrix == "thermoplastic":
        return 100.0 if "unsized" in sizing or "thermoplastic" in compatibility else 45.0

    if matrix in compatibility:
        return 100.0

    if matrix == "epoxy" and any(s in sizing for s in ["gp", "gs", " g", "g"]):
        return 95.0

    return 60.0


def certification_fit(row: pd.Series, required: bool) -> float:
    if not required:
        return 100.0

    cert = norm(row.get("certification", ""))
    return 100.0 if cert and cert not in {"not stated", "nan"} else 0.0


def engineering_score(row: pd.Series, req: Requirements, app: dict) -> float:
    values = {
        "strength": ratio_score(row["tensile_strength_mpa"], req.strength_mpa, True),
        "modulus": ratio_score(row["tensile_modulus_gpa"], req.modulus_gpa, True),
        "elongation": ratio_score(row["ultimate_elongation_percent"], req.elongation_percent, True),
        "density": ratio_score(row["fiber_density_g_cm3"], req.max_density_g_cm3, False),
    }

    weights = {
        key: val / sum(app["importance"].values())
        for key, val in app["importance"].items()
    }
    return weighted_mean(values, weights)



def _contains_product(value: Any, product: str) -> bool:
    value_n = norm(value)
    product_n = norm(product)
    return bool(product_n) and (
        value_n == product_n or product_n in value_n or value_n in product_n
    )


def compatibility_matches(
    fiber_row: pd.Series,
    compatibility: pd.DataFrame,
    process: Optional[str],
    prepreg_row: Optional[pd.Series] = None,
    resin_row: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Match a complete Fiber + Prepreg + Resin + Process system against
    documented compatibility evidence.

    Missing evidence is returned as UNKNOWN rather than treated as
    incompatibility.
    """
    manufacturer = norm(fiber_row.get("manufacturer", ""))
    fiber_product = clean_text(fiber_row.get("product", ""))
    fiber_id = norm(fiber_row.get("fiber_id", ""))

    matches = compatibility.copy()

    # Fiber identity: prefer exact ID, otherwise manufacturer + product.
    # Use the original helper functions directly; no separate normalization
    # pass is needed.
    manufacturer_values = matches.get(
        "manufacturer", pd.Series("", index=matches.index)
    ).fillna("").astype(str).map(norm)

    fiber_product_values = matches.get(
        "fiber_product", pd.Series("", index=matches.index)
    ).fillna("").astype(str).map(norm)

    if fiber_id and "fiber_id" in matches.columns:
        id_match = matches["fiber_id"].fillna("").astype(str).map(norm).eq(fiber_id)
        text_match = (
            manufacturer_values.eq(manufacturer)
            & fiber_product_values.apply(
                lambda x: _contains_product(x, fiber_product)
            )
        )
        matches = matches.loc[id_match | text_match].copy()
    else:
        matches = matches.loc[
            manufacturer_values.eq(manufacturer)
            & fiber_product_values.apply(
                lambda x: _contains_product(x, fiber_product)
            )
        ].copy()

    if process and not matches.empty:
        process_values = matches.get(
            "process", pd.Series("", index=matches.index)
        ).fillna("").astype(str).map(norm)
        matches = matches.loc[
            process_values.eq(norm(process))
        ].copy()

    # Prepreg identity.
    if prepreg_row is not None and not matches.empty:
        prepreg_product = clean_text(
            prepreg_row.get("product", prepreg_row.get("prepreg_product", ""))
        )
        if prepreg_product:
            prepreg_values = matches.get(
                "prepreg_product", pd.Series("", index=matches.index)
            ).fillna("").astype(str).map(norm)
            matches = matches.loc[
                prepreg_values.apply(
                    lambda x: _contains_product(x, prepreg_product)
                )
            ].copy()

    # Resin identity.
    if resin_row is not None and not matches.empty:
        resin_product = clean_text(
            resin_row.get("product", resin_row.get("resin_product", ""))
        )
        if resin_product:
            resin_values = matches.get(
                "resin_product", pd.Series("", index=matches.index)
            ).fillna("").astype(str).map(norm)
            matches = matches.loc[
                resin_values.apply(
                    lambda x: _contains_product(x, resin_product)
                )
            ].copy()

    return matches


def compatibility_fit(
    fiber_row: pd.Series,
    compatibility: pd.DataFrame,
    process: Optional[str],
    prepreg_row: Optional[pd.Series] = None,
    resin_row: Optional[pd.Series] = None,
) -> dict:
    """Return the strongest evidence for the complete material system."""
    matches = compatibility_matches(
        fiber_row, compatibility, process, prepreg_row, resin_row
    )

    if matches.empty:
        return {
            "score": np.nan,
            "level": 0,
            "status": "UNKNOWN",
            "compatibility_id": "",
            "prepreg_product": "",
            "resin_product": "",
            "process": process or "",
            "evidence_type": "",
            "source": "",
            "manufacturer_statement": "",
        }

    best = matches.sort_values("compatibility_level", ascending=False).iloc[0]
    raw_level = number(best.get("compatibility_level", 0))
    level = int(raw_level) if np.isfinite(raw_level) else 0

    return {
        "score": min(level / 4.0, 1.0) * 100.0,
        "level": level,
        "status": "EXPLICIT" if level >= 3 else "SUPPORTED",
        "compatibility_id": clean_text(best.get("compatibility_id", "")),
        "prepreg_product": clean_text(best.get("prepreg_product", "")),
        "resin_product": clean_text(best.get("resin_product", "")),
        "process": clean_text(best.get("process", "")),
        "evidence_type": clean_text(best.get("evidence_type", "")),
        "source": clean_text(best.get("source", "")),
        "manufacturer_statement": clean_text(best.get("manufacturer_statement", "")),
    }


def _candidate_rows(
    prepregs: pd.DataFrame,
    resins: pd.DataFrame,
    compatibility: pd.DataFrame,
    fiber: pd.Series,
    req: Requirements,
) -> Iterable[tuple[pd.Series, pd.Series]]:
    """
    Generate prepreg/resin pairs from documented compatibility evidence.

    This preserves the v6 architecture while avoiding a blind Cartesian
    product of every prepreg with every resin.
    """
    if compatibility.empty:
        return

    manufacturer = norm(fiber.get("manufacturer", ""))
    fiber_product = clean_text(fiber.get("product", ""))
    fiber_id = norm(fiber.get("fiber_id", ""))

    matches = compatibility.copy()

    manufacturer_values = matches.get(
        "manufacturer", pd.Series("", index=matches.index)
    ).fillna("").astype(str).map(norm)

    fiber_product_values = matches.get(
        "fiber_product", pd.Series("", index=matches.index)
    ).fillna("").astype(str).map(norm)

    if fiber_id and "fiber_id" in matches.columns:
        id_match = matches["fiber_id"].fillna("").astype(str).map(norm).eq(fiber_id)
        text_match = (
            manufacturer_values.eq(manufacturer)
            & fiber_product_values.apply(
                lambda x: _contains_product(x, fiber_product)
            )
        )
        matches = matches.loc[id_match | text_match].copy()
    else:
        matches = matches.loc[
            manufacturer_values.eq(manufacturer)
            & fiber_product_values.apply(
                lambda x: _contains_product(x, fiber_product)
            )
        ].copy()

    if req.process and "process" in matches.columns and not matches.empty:
        process_values = matches["process"].fillna("").astype(str).map(norm)
        matches = matches.loc[
            process_values.eq(norm(req.process))
        ].copy()

    if matches.empty:
        return

    yielded = set()

    for _, evidence in matches.iterrows():
        evidence_prepreg = clean_text(evidence.get("prepreg_product", ""))
        evidence_resin = clean_text(evidence.get("resin_product", ""))

        prep_candidates = prepregs
        if evidence_prepreg:
            prep_values = prepregs.get(
                "product", pd.Series("", index=prepregs.index)
            ).fillna("").astype(str).map(norm)

            prep_candidates = prepregs.loc[
                prep_values.apply(
                    lambda x: _contains_product(x, evidence_prepreg)
                )
            ]

        resin_candidates = resins
        if evidence_resin:
            resin_values = resins.get(
                "product", pd.Series("", index=resins.index)
            ).fillna("").astype(str).map(norm)

            resin_candidates = resins.loc[
                resin_values.apply(
                    lambda x: _contains_product(x, evidence_resin)
                )
            ]

        for prep_idx, prepreg in prep_candidates.iterrows():
            for resin_idx, resin in resin_candidates.iterrows():
                key = (prep_idx, resin_idx)

                if key not in yielded:
                    yielded.add(key)
                    yield prepreg, resin
def build_system_ranked_results(
    fibers: pd.DataFrame,
    prepregs: pd.DataFrame,
    resins: pd.DataFrame,
    compatibility: pd.DataFrame,
    req: Requirements,
    app: dict,
) -> pd.DataFrame:
    """
    Build complete material-system recommendations:

        Requirements
             ↓
        Fiber candidates
             ↓
        Prepreg candidates
             ↓
        Resin candidates
             ↓
        Compatibility evidence
             ↓
        Complete systems
    """
    fiber_candidates = numerical_filter(fibers, req)

    if req.require_certification and "certification" in fiber_candidates.columns:
        cert_mask = (
            fiber_candidates["certification"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )
        fiber_candidates = fiber_candidates.loc[cert_mask].copy()

    records = []

    for _, fiber in fiber_candidates.iterrows():
        eng = engineering_score(fiber, req, app)
        app_score = application_fit(fiber, app)
        grade_score_value = grade_fit(fiber, app["name"])

        candidate_pairs = list(
            _candidate_rows(
                prepregs, resins, compatibility, fiber, req
            )
        )

        for prepreg, resin in candidate_pairs:
            proc = process_fit(fiber, req.process)
            matrix = fiber_matrix_fit(fiber, req.matrix)
            cert = certification_fit(fiber, req.require_certification)

            compat = compatibility_fit(
                fiber,
                compatibility,
                req.process,
                prepreg_row=prepreg,
                resin_row=resin,
            )

            # A complete system gets a compatibility bonus only when
            # documented evidence exists. Unknown remains neutral.
            if np.isfinite(compat["score"]):
                final = (
                    0.60 * eng +
                    0.07 * app_score +
                    0.05 * grade_score_value +
                    0.07 * proc +
                    0.05 * matrix +
                    0.02 * cert +
                    0.14 * compat["score"]
                )
            else:
                final = (
                    0.70 * eng +
                    0.08 * app_score +
                    0.06 * grade_score_value +
                    0.08 * proc +
                    0.06 * matrix +
                    0.02 * cert
                )

            records.append({
                "Fiber_ID": fiber.get("fiber_id", ""),
                "Manufacturer": fiber.get("manufacturer", ""),
                "Fiber": fiber.get("product", ""),
                "Strength_MPa": fiber.get("tensile_strength_mpa", np.nan),
                "Modulus_GPa": fiber.get("tensile_modulus_gpa", np.nan),
                "Density_g_cm3": fiber.get("fiber_density_g_cm3", np.nan),
                "Elongation_percent": fiber.get("ultimate_elongation_percent", np.nan),
                "Prepreg": prepreg.get("product", prepreg.get("prepreg_product", "")),
                "Resin": resin.get("product", resin.get("resin_product", "")),
                "Process": req.process or "",
                "Engineering_Score": round(eng, 2),
                "Application_Fit": round(app_score, 2),
                "Grade_Fit": round(grade_score_value, 2),
                "Process_Fit": round(proc, 2),
                "Matrix_Fit": round(matrix, 2),
                "Certification_Fit": round(cert, 2),
                "Compatibility_Status": compat["status"],
                "Compatibility_Level": compat["level"],
                "Compatibility_ID": compat["compatibility_id"],
                "Compatibility_Evidence": compat["evidence_type"],
                "Compatibility_Source": compat["source"],
                "Manufacturer_Statement": compat["manufacturer_statement"],
                "Final_Score": round(final, 2),
                "Source_File": fiber.get("source_file", ""),
                "Source_URL": fiber.get("source_url", ""),
                "Data_Status": fiber.get("fiber_data_status", ""),
                "Fiber_Class": fiber.get("fiber_class", ""),
                "Tow_or_Sizing": fiber.get(
                    "sizing_options",
                    fiber.get("sizing_compatibility", "")
                ),
                "Certification": fiber.get("certification", ""),
                "Compatible_Prepreg": compat["prepreg_product"],
                "Compatible_Resin": compat["resin_product"],
            })

    if not records:
        return pd.DataFrame()

    result = (
        pd.DataFrame(records)
        .sort_values(["Compatibility_Level", "Final_Score"], ascending=[False, False])
        .reset_index(drop=True)
    )

    result["Rank"] = np.arange(1, len(result) + 1)

    if not result.empty:
        max_score = pd.to_numeric(
            result["Final_Score"], errors="coerce"
        ).max()

        if np.isfinite(max_score) and max_score > 0:
            result["Relative_Score_percent"] = (
                100.0 * result["Final_Score"] / max_score
            ).round(1)
        else:
            result["Relative_Score_percent"] = np.nan

    return result

def print_results(result: pd.DataFrame, req: Requirements, app: dict) -> None:
    print("\n" + "=" * 70)
    print("TOP COMPLETE MATERIAL SYSTEMS")
    print("=" * 70)

    print(
        f"\nApplication: {app['name']}\n"
        f"Process: {req.process or 'None'}\n"
        f"Matrix: {req.matrix or 'None'}"
    )

    if result.empty:
        print("\nNO CANDIDATE SATISFIES THE HARD REQUIREMENTS.")
        return

    for _, r in result.head(5).iterrows():
        score = r.get("Relative_Score_percent", np.nan)
        score_text = f"{float(score):.1f}%" if pd.notna(score) else "N/A"

        print(
            f"\n#{int(r['Rank'])} {r.get('Manufacturer', '')} "
            f"{r.get('Fiber', '')} — {score_text}"
        )

        print(
            f"  Strength: {r['Strength_MPa']:.0f} MPa | "
            f"Modulus: {r['Modulus_GPa']:.0f} GPa | "
            f"Density: {r['Density_g_cm3']:.2f} g/cm³ | "
            f"Elongation: {r['Elongation_percent']:.1f}%"
        )

        print(
            f"  System: {r.get('Fiber', '')} + "
            f"{r.get('Prepreg', '')} + {r.get('Resin', '')} | "
            f"Process: {r.get('Process', '') or 'Not specified'}"
        )

        print(
            f"  Engineering: {r['Engineering_Score']:.1f} | "
            f"Application: {r['Application_Fit']:.1f} | "
            f"Grade: {r['Grade_Fit']:.1f} | "
            f"Process: {r['Process_Fit']:.1f} | "
            f"Matrix: {r['Matrix_Fit']:.1f}"
        )

        fiber_class = r.get("Fiber_Class", "")
        sizing = r.get("Tow_or_Sizing", "")
        certification = r.get("Certification", "")

        if fiber_class:
            print(f"  Fiber class: {fiber_class}")
        if sizing:
            print(f"  Sizing / tow: {sizing}")

        print(f"  Certification: {certification or 'Not stated'}")

        status = r.get("Compatibility_Status", "UNKNOWN")

        if status == "UNKNOWN":
            print(
                "  Compatibility: UNKNOWN — no documented match "
                "in Compatibility_Master_v1"
            )
        else:
            level = number(r.get("Compatibility_Level", 0))
            level_text = str(int(level)) if np.isfinite(level) else "0"

            print(
                f"  Compatibility: {status} | "
                f"Level {level_text} | "
                f"Prepreg: {r.get('Compatible_Prepreg', r.get('Prepreg', '')) or 'Not stated'}"
            )
            print(
                f"  Resin: {r.get('Compatible_Resin', r.get('Resin', '')) or 'Not stated'}"
            )
            print(
                f"  Evidence: {r.get('Compatibility_Evidence', '') or 'Not stated'}"
            )
            print(
                f"  Source: {r.get('Compatibility_Source', '') or 'Not stated'}"
            )


def ask_float(prompt: str, minimum: float = 0.0) -> float:
    while True:
        try:
            value = float(input(prompt))
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"Enter a number >= {minimum}.")


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Enter yes or no.")


def main() -> None:
    print("\n" + "=" * 70)
    print(" COMPOSITE MATERIAL RECOMMENDATION ENGINE v6")
    print("=" * 70)

    fibers, prepregs, resins, compatibility = load_data()
    print(f"\nLoaded {len(fibers)} carbon-fiber grades.")
    print(f"Loaded {len(prepregs)} prepreg records.")
    print(f"Loaded {len(resins)} resin records.")
    print(f"Loaded {len(compatibility)} compatibility records.")

    print("\nAPPLICATION")
    for key, value in APPLICATIONS.items():
        print(f"{key}. {value['name']}")

    choice = input("Choose application: ").strip()
    while choice not in APPLICATIONS:
        choice = input("Choose a valid option: ").strip()

    app = APPLICATIONS[choice].copy()

    if choice == "6":
        print("\nCUSTOM IMPORTANCE (1-10)")
        app["importance"] = {
            "strength": ask_float("Strength importance: ", 1),
            "modulus": ask_float("Modulus importance: ", 1),
            "density": ask_float("Density importance: ", 1),
            "elongation": ask_float("Elongation importance: ", 1),
        }

    print("\nENGINEERING REQUIREMENTS")
    req = Requirements(
        strength_mpa=ask_float("Minimum tensile strength (MPa): "),
        modulus_gpa=ask_float("Minimum tensile modulus (GPa): "),
        max_density_g_cm3=ask_float("Maximum density (g/cm³): "),
        elongation_percent=ask_float("Minimum elongation (%): "),
        hard_strength=ask_yes_no("Strength is HARD? "),
        hard_modulus=ask_yes_no("Modulus is HARD? "),
        hard_density=ask_yes_no("Density is HARD? "),
        hard_elongation=ask_yes_no("Elongation is HARD? "),
        process=None,
        matrix=None,
        require_certification=False,
    )

    print("\nPROCESS")
    for key, name in PROCESS_OPTIONS.items():
        print(f"{key}. {name or 'No process preference'}")
    req.process = PROCESS_OPTIONS.get(input("Choose process: ").strip())

    print("\nMATRIX")
    for key, name in MATRIX_OPTIONS.items():
        print(f"{key}. {name or 'No matrix preference'}")
    req.matrix = MATRIX_OPTIONS.get(input("Choose matrix: ").strip())

    req.require_certification = ask_yes_no(
        "\nRequire an explicitly stated certification? "
    )

    result = build_system_ranked_results(
        fibers, prepregs, resins, compatibility, req, app
    )

    print_results(result, req, app)

    if not result.empty:
        export = ask_yes_no("\nExport complete ranking to CSV? ")
        if export:
            out = BASE_DIR / "recommendation_results_v6.csv"
            result.to_csv(out, index=False)
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()