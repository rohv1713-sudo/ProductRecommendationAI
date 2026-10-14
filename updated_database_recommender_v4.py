"""
Composite Material Recommendation Engine v4

Refactor of the earlier Hexcel-only recommender.

Key changes:
- Uses the current combined carbon-fiber master database.
- Normalizes the database schema once, instead of hard-coding older column names.
- Keeps HARD constraints separate from PREFERRED scoring.
- Uses bounded 0-100 component scores so final ranking is interpretable.
- Keeps manufacturer/source traceability in the result.
- Adds optional prepreg/resin process filtering without pretending
  generic compatibility is a manufacturer qualification.
- Leaves room for future laminate/process recommendation layers.

Database files expected beside this script (or set paths below):
    Carbon_Fiber_Master_v6_Top5.csv
    Prepreg_Master_Database_v2_Expanded.csv
    Resin_Master_Database_v10_HiFlow_Added.csv

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

# ============================================================
# DATABASE PATHS
# Each database is independent.
#
# Fiber recommendation reads ONLY the fiber database.
# Prepreg/resin databases are read separately by their own loaders.
# ============================================================

FIBER_FILE = BASE_DIR / "Carbon_Fiber_Master_v6_Top5(1).csv"
PREPREG_FILE = BASE_DIR / "Prepreg_Master_Database_v2_Expanded.csv"
RESIN_FILE = BASE_DIR / "Resin_Master_Database_v10.csv"


def resolve_file(primary: Path, description: str) -> Path:
    """Resolve one database file independently."""
    if primary.exists():
        return primary

    fallback = Path("/mnt/data") / primary.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"{description} was not found. Expected:\n"
        f"  {primary}\n"
        f"or:\n"
        f"  {fallback}\n"
    )


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


# Composite manufacturing processes. These are deliberately separate
# from fiber-format operations such as weaving or braiding.
PROCESS_OPTIONS = {
    "1": None,
    "2": "VARTM / infusion",
    "3": "RTM",
    "4": "autoclave",
    "5": "OOA / VBO",
    "6": "press / compression molding",
    "7": "filament winding",
    "8": "pultrusion",
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


def load_fiber_database() -> pd.DataFrame:
    """Read ONLY the carbon-fiber database."""
    path = resolve_file(FIBER_FILE, "Carbon-fiber database")
    fibers = pd.read_csv(path)
    return normalize_fiber_db(fibers)


def load_prepreg_database() -> pd.DataFrame:
    """Read ONLY the prepreg database."""
    path = resolve_file(PREPREG_FILE, "Prepreg database")
    prepregs = pd.read_csv(path)
    return normalize_prepreg_db(prepregs)


def load_resin_database() -> pd.DataFrame:
    """Read ONLY the resin database."""
    path = resolve_file(RESIN_FILE, "Resin database")
    resins = pd.read_csv(path)
    return normalize_resin_db(resins)


def load_data() -> pd.DataFrame:
    """
    Backward-compatible wrapper.
    Current recommender uses ONLY the fiber database.
    """
    return load_fiber_database()


def normalize_fiber_db(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [
        "fiber_id", "manufacturer", "product",
        "tensile_strength_mpa", "tensile_modulus_gpa",
        "ultimate_elongation_percent", "fiber_density_g_cm3",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Fiber database is missing columns: {missing}")

    numeric = [
        "tensile_strength_mpa",
        "tensile_modulus_gpa",
        "ultimate_elongation_percent",
        "fiber_density_g_cm3",
        "filament_diameter_um",
        "carbon_content_percent",
    ]
    for col in numeric:
        if col in out:
            out[col] = out[col].apply(number)

    return out


def normalize_prepreg_db(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "fiber_areal_weight_gsm",
        "resin_content_percent",
        "nominal_cured_ply_thickness_mm",
        "nominal_fiber_volume_percent",
        "nominal_laminate_density_g_cm3",
        "cure_temperature_C",
        "Tg_C",
        "service_temperature_C",
    ]:
        if col in out:
            out[col] = out[col].apply(number)

    return out


def normalize_resin_db(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "density_g_cm3",
        "viscosity_value",
        "viscosity_temperature_C",
        "cure_temperature_C",
        "Tg_C",
        "service_temperature_C",
    ]:
        if col in out:
            out[col] = out[col].apply(number)

    return out


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


def constraint_diagnostics(df: pd.DataFrame, req: Requirements) -> list[dict]:
    """
    Diagnose every numerical requirement independently.

    This is intentionally separate from numerical_filter(): a failed
    candidate set should tell the user WHICH hard constraint caused it,
    rather than only reporting that no candidate survived.
    """
    checks = [
        (
            "Strength",
            "tensile_strength_mpa",
            req.strength_mpa,
            "≥",
            lambda s, x: s.notna() & (s >= x),
        ),
        (
            "Modulus",
            "tensile_modulus_gpa",
            req.modulus_gpa,
            "≥",
            lambda s, x: s.notna() & (s >= x),
        ),
        (
            "Density",
            "fiber_density_g_cm3",
            req.max_density_g_cm3,
            "≤",
            lambda s, x: s.notna() & (s <= x),
        ),
        (
            "Elongation",
            "ultimate_elongation_percent",
            req.elongation_percent,
            "≥",
            lambda s, x: s.notna() & (s >= x),
        ),
    ]

    diagnostics = []

    for name, column, threshold, operator, test in checks:
        series = pd.to_numeric(df[column], errors="coerce")
        valid = series.notna()
        passed = test(series, threshold)
        count = int(passed.sum())
        missing = int((~valid).sum())

        diagnostics.append({
            "requirement": name,
            "operator": operator,
            "threshold": threshold,
            "hard": getattr(req, f"hard_{name.lower()}"),
            "passing_candidates": count,
            "missing_data": missing,
            "total_candidates": len(df),
        })

    return diagnostics


def print_constraint_diagnostics(df: pd.DataFrame, req: Requirements) -> None:
    """Print a readable explanation of which constraints eliminate candidates."""
    diagnostics = constraint_diagnostics(df, req)

    print("\n" + "=" * 70)
    print("CONSTRAINT DIAGNOSTICS")
    print("=" * 70)

    for d in diagnostics:
        status = "✓" if d["passing_candidates"] > 0 else "✗"

        if d["requirement"] == "Strength":
            unit = "MPa"
        elif d["requirement"] == "Modulus":
            unit = "GPa"
        elif d["requirement"] == "Density":
            unit = "g/cm³"
        else:
            unit = "%"

        hard_text = "HARD" if d["hard"] else "PREFERRED"

        print(
            f"{status} {d['requirement']:<12} "
            f"{d['operator']} {d['threshold']:.2f} {unit:<7} "
            f"| {d['passing_candidates']:>3}/{d['total_candidates']} pass "
            f"| {hard_text}"
        )

        if d["missing_data"]:
            print(
                f"    Missing data: {d['missing_data']} candidate(s) "
                f"— missing values only eliminate candidates when this "
                f"requirement is HARD."
            )

    hard_failures = [
        d for d in diagnostics
        if d["hard"] and d["passing_candidates"] == 0
    ]

    if hard_failures:
        print("\nWHY NO CANDIDATE QUALIFIED:")
        for d in hard_failures:
            if d["requirement"] == "Density":
                explanation = (
                    "The maximum allowed density is below the density of "
                    "every candidate with published density data."
                )
            else:
                explanation = (
                    f"No candidate satisfies the HARD {d['requirement'].lower()} "
                    f"requirement."
                )
            print(f"  • {d['requirement']}: {explanation}")

        print("\nSUGGESTION:")
        print(
            "Relax one of the constraints marked ✗, change it from HARD "
            "to PREFERRED, or verify that the requested value/unit is correct."
        )
    else:
        print("\nAll HARD constraints have at least one qualifying candidate.")


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


def text_matches(value: Any, terms: Iterable[str]) -> bool:
    text = norm(value)
    return any(term.lower() in text for term in terms)


def process_database_score(value: Any, process: Optional[str]) -> float:
    """Score only explicit process-suitability text; no inferred qualification."""
    if not process:
        return 100.0

    text = norm(value)

    aliases = {
        "VARTM / infusion": ["vartm", "vacuum infusion", "infusion", "lri"],
        "RTM": ["rtm", "resin transfer molding"],
        "autoclave": ["autoclave"],
        "OOA / VBO": ["ooa", "out-of-autoclave", "vbo", "vacuum bag only"],
        "press / compression molding": ["press", "compression molding"],
        "filament winding": ["filament winding"],
        "pultrusion": ["pultrusion"],
    }

    terms = aliases.get(process, [process])
    if any(term in text for term in terms):
        return 100.0

    # Unknown/non-explicit suitability is not treated as incompatible.
    return 50.0


def matrix_database_score(value: Any, matrix: Optional[str]) -> float:
    """Score explicit matrix-family information only."""
    if not matrix:
        return 100.0

    text = norm(value)

    aliases = {
        "epoxy": ["epoxy"],
        "thermoplastic": ["thermoplastic", "peek", "pps", "pa", "pekk"],
        "phenolic": ["phenolic"],
        "vinyl ester": ["vinyl ester"],
        "cyanate ester": ["cyanate ester", "cyanate"],
        "BMI": ["bmi", "bismaleimide"],
        "polyimide": ["polyimide"],
        "polycarbonate": ["polycarbonate"],
    }

    terms = aliases.get(matrix, [matrix])
    if any(term in text for term in terms):
        return 100.0

    return 50.0


def certification_database_score(value: Any, required: bool) -> float:
    if not required:
        return 100.0

    text = norm(value)
    if text and text not in {"nan", "not stated"}:
        return 100.0

    return 0.0


def build_prepreg_results(
    prepregs: pd.DataFrame,
    req: Requirements,
    selected_fibers: pd.DataFrame,
) -> pd.DataFrame:
    """Rank prepreg systems independently from the fiber database."""
    if prepregs.empty:
        return pd.DataFrame()

    candidates = prepregs.copy()

    # Hard process/matrix filters are applied only when the database explicitly
    # states suitability. Unknown does not become a false incompatibility.
    if req.process:
        explicit = candidates["process_suitability"].fillna("").astype(str)
        pass_mask = explicit.apply(lambda x: process_database_score(x, req.process) == 100.0)
        known_mask = explicit.str.strip().ne("")
        candidates = candidates.loc[(~known_mask) | pass_mask].copy()

    if req.matrix:
        explicit = candidates["matrix_family"].fillna("").astype(str)
        pass_mask = explicit.apply(lambda x: matrix_database_score(x, req.matrix) == 100.0)
        known_mask = explicit.str.strip().ne("")
        candidates = candidates.loc[(~known_mask) | pass_mask].copy()

    if req.require_certification:
        cert = candidates["qualification_certification"].fillna("").astype(str)
        candidates = candidates.loc[cert.str.strip().ne("")].copy()

    if candidates.empty:
        return pd.DataFrame()

    selected_products = set(
        norm(x) for x in selected_fibers.get("Product", pd.Series(dtype=str)).tolist()
    )
    selected_ids = set(
        norm(x) for x in selected_fibers.get("Fiber_ID", pd.Series(dtype=str)).tolist()
    )

    records = []
    for _, row in candidates.iterrows():
        fiber_grade = norm(row.get("fiber_grade", ""))
        fiber_mfr = norm(row.get("fiber_manufacturer", ""))

        exact_match = any(
            fiber_grade and (fiber_grade in sid or fiber_grade in product)
            for sid, product in zip(selected_ids, selected_products)
        )

        proc_score = process_database_score(row.get("process_suitability", ""), req.process)
        matrix_score = matrix_database_score(row.get("matrix_family", ""), req.matrix)
        cert_score = certification_database_score(
            row.get("qualification_certification", ""), req.require_certification
        )

        # Explicit fiber-grade match is rewarded; otherwise the system remains
        # a standalone prepreg recommendation rather than claiming compatibility.
        fiber_match_score = 100.0 if exact_match else 50.0

        final = (
            0.40 * proc_score +
            0.30 * matrix_score +
            0.20 * fiber_match_score +
            0.10 * cert_score
        )

        records.append({
            "Rank": 0,
            "Prepreg_ID": row.get("prepreg_id", ""),
            "Manufacturer": row.get("manufacturer", ""),
            "Product_System": row.get("product_system", ""),
            "Matrix": row.get("matrix_family", ""),
            "Fiber_Manufacturer": row.get("fiber_manufacturer", ""),
            "Fiber_Grade": row.get("fiber_grade", ""),
            "Fiber_Form": row.get("fiber_form", ""),
            "FAW_gsm": row.get("fiber_areal_weight_gsm", np.nan),
            "Resin_Content_percent": row.get("resin_content_percent", np.nan),
            "Ply_Thickness_mm": row.get("nominal_cured_ply_thickness_mm", np.nan),
            "Vf_percent": row.get("nominal_fiber_volume_percent", np.nan),
            "Cure_Temperature_C": row.get("cure_temperature_C", np.nan),
            "Tg_C": row.get("Tg_C", np.nan),
            "Process_Suitability": row.get("process_suitability", ""),
            "Qualification": row.get("qualification_certification", ""),
            "Fiber_Match": "EXPLICIT" if exact_match else "NOT ESTABLISHED",
            "Process_Score": round(proc_score, 2),
            "Matrix_Score": round(matrix_score, 2),
            "Fiber_Match_Score": round(fiber_match_score, 2),
            "Certification_Score": round(cert_score, 2),
            "Final_Score": round(final, 2),
            "Source_Document": row.get("source_document", ""),
            "Source_URL": row.get("source_url", ""),
            "Data_Status": row.get("data_status", ""),
        })

    result = pd.DataFrame(records).sort_values(
        ["Final_Score", "Fiber_Match_Score"], ascending=False
    ).reset_index(drop=True)

    result["Rank"] = np.arange(1, len(result) + 1)
    return result


def build_resin_results(resins: pd.DataFrame, req: Requirements) -> pd.DataFrame:
    """Rank resin systems independently from fiber and prepreg databases."""
    if resins.empty:
        return pd.DataFrame()

    candidates = resins.copy()

    if req.process:
        explicit = candidates["process_suitability"].fillna("").astype(str)
        pass_mask = explicit.apply(lambda x: process_database_score(x, req.process) == 100.0)
        known_mask = explicit.str.strip().ne("")
        candidates = candidates.loc[(~known_mask) | pass_mask].copy()

    if req.matrix:
        explicit = candidates["resin_family"].fillna("").astype(str)
        pass_mask = explicit.apply(lambda x: matrix_database_score(x, req.matrix) == 100.0)
        known_mask = explicit.str.strip().ne("")
        candidates = candidates.loc[(~known_mask) | pass_mask].copy()

    if req.require_certification:
        cert = candidates["qualification_certification"].fillna("").astype(str)
        candidates = candidates.loc[cert.str.strip().ne("")].copy()

    if candidates.empty:
        return pd.DataFrame()

    records = []
    for _, row in candidates.iterrows():
        proc_score = process_database_score(row.get("process_suitability", ""), req.process)
        matrix_score = matrix_database_score(row.get("resin_family", ""), req.matrix)
        cert_score = certification_database_score(
            row.get("qualification_certification", ""), req.require_certification
        )

        final = 0.50 * proc_score + 0.40 * matrix_score + 0.10 * cert_score

        records.append({
            "Rank": 0,
            "Resin_ID": row.get("resin_id", ""),
            "Manufacturer": row.get("manufacturer", ""),
            "Product_System": row.get("product_system", ""),
            "Resin_Family": row.get("resin_family", ""),
            "Process_Suitability": row.get("process_suitability", ""),
            "Cure_Type": row.get("cure_type", ""),
            "Viscosity": row.get("viscosity_value", np.nan),
            "Viscosity_Unit": row.get("viscosity_unit", ""),
            "Cure_Temperature_C": row.get("cure_temperature_C", np.nan),
            "Tg_C": row.get("Tg_C", np.nan),
            "Service_Temperature_C": row.get("service_temperature_C", np.nan),
            "Qualification": row.get("qualification_certification", ""),
            "Process_Score": round(proc_score, 2),
            "Matrix_Score": round(matrix_score, 2),
            "Certification_Score": round(cert_score, 2),
            "Final_Score": round(final, 2),
            "Source_Document": row.get("source_document", ""),
            "Source_URL": row.get("source_url", ""),
            "Data_Status": row.get("data_status", ""),
        })

    result = pd.DataFrame(records).sort_values("Final_Score", ascending=False).reset_index(drop=True)
    result["Rank"] = np.arange(1, len(result) + 1)
    return result


def print_system_database_results(
    prepreg_results: pd.DataFrame,
    resin_results: pd.DataFrame,
) -> None:
    print("\n" + "=" * 70)
    print("PREPREG DATABASE RECOMMENDATIONS")
    print("=" * 70)

    if prepreg_results.empty:
        print("No prepreg candidates match the selected process/matrix filters.")
    else:
        for _, r in prepreg_results.head(5).iterrows():
            print(
                f"#{int(r['Rank'])} {r['Manufacturer']} {r['Product_System']} "
                f"— {r['Final_Score']:.1f}/100"
            )
            print(
                f"  Matrix: {r['Matrix']} | Fiber: {r['Fiber_Manufacturer']} "
                f"{r['Fiber_Grade']} | Fiber match: {r['Fiber_Match']}"
            )
            print(
                f"  Process: {r['Process_Suitability']} | "
                f"Cure: {r['Cure_Temperature_C']}°C | Tg: {r['Tg_C']}"
            )

    print("\n" + "=" * 70)
    print("RESIN DATABASE RECOMMENDATIONS")
    print("=" * 70)

    if resin_results.empty:
        print("No resin candidates match the selected process/matrix filters.")
    else:
        for _, r in resin_results.head(5).iterrows():
            print(
                f"#{int(r['Rank'])} {r['Manufacturer']} {r['Product_System']} "
                f"— {r['Final_Score']:.1f}/100"
            )
            print(
                f"  Family: {r['Resin_Family']} | Process: {r['Process_Suitability']} | "
                f"Cure: {r['Cure_Temperature_C']}°C | Tg: {r['Tg_C']}"
            )
            print(
                f"  Viscosity: {r['Viscosity']} {r['Viscosity_Unit']} | "
                f"Qualification: {r['Qualification'] or 'Not stated'}"
            )


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


def build_ranked_results(df: pd.DataFrame, req: Requirements, app: dict) -> pd.DataFrame:
    candidates = numerical_filter(df, req)

    if req.require_certification:
        cert_mask = candidates["certification"].fillna("").astype(str).str.strip().ne("")
        candidates = candidates.loc[cert_mask].copy()

    if candidates.empty:
        return pd.DataFrame()

    records = []
    for _, row in candidates.iterrows():
        eng = engineering_score(row, req, app)
        app_score = application_fit(row, app)
        grade_score_value = grade_fit(row, app["name"])
        proc = process_fit(row, req.process)
        matrix = fiber_matrix_fit(row, req.matrix)
        cert = certification_fit(row, req.require_certification)

        final = (
            0.70 * eng +
            0.08 * app_score +
            0.06 * grade_score_value +
            0.08 * proc +
            0.06 * matrix +
            0.02 * cert
        )

        records.append({
            "Fiber_ID": row["fiber_id"],
            "Manufacturer": row["manufacturer"],
            "Product": row["product"],
            "Strength_MPa": row["tensile_strength_mpa"],
            "Modulus_GPa": row["tensile_modulus_gpa"],
            "Density_g_cm3": row["fiber_density_g_cm3"],
            "Elongation_percent": row["ultimate_elongation_percent"],
            "Tow_or_Sizing": row.get("sizing_options", ""),
            "Fiber_Class": row.get("fiber_class", ""),
            "Certification": row.get("certification", ""),
            "Engineering_Score": round(eng, 2),
            "Application_Fit": round(app_score, 2),
            "Grade_Fit": round(grade_score_value, 2),
            "Process_Fit": round(proc, 2),
            "Matrix_Fit": round(matrix, 2),
            "Certification_Fit": round(cert, 2),
            "Final_Score": round(final, 2),
            "Source_File": row.get("source_file", ""),
            "Source_URL": row.get("source_url", ""),
            "Data_Status": row.get("fiber_data_status", ""),
        })

    result = pd.DataFrame(records).sort_values("Final_Score", ascending=False).reset_index(drop=True)

    if not result.empty:
        result["Rank"] = np.arange(1, len(result) + 1)
        result["Relative_Score_percent"] = (
            100.0 * result["Final_Score"] / result["Final_Score"].max()
        ).round(1)

    return result


def print_results(result: pd.DataFrame, req: Requirements, app: dict, all_fibers: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TOP FIBER RECOMMENDATIONS")
    print("=" * 70)

    print(
        f"\nApplication: {app['name']}\n"
        f"Process: {req.process or 'None'}\n"
        f"Matrix: {req.matrix or 'None'}"
    )

    if result.empty:
        print("\nNO CANDIDATE SATISFIES THE HARD REQUIREMENTS.")
        print_constraint_diagnostics(all_fibers, req)
        return

    for _, r in result.head(5).iterrows():
        print(
            f"\n#{int(r['Rank'])} {r['Manufacturer']} {r['Product']} "
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
        print(f"  Fiber class: {r['Fiber_Class']}")
        print(f"  Sizing: {r['Tow_or_Sizing']}")
        print(f"  Certification: {r['Certification'] or 'Not stated'}")


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
    print(" COMPOSITE MATERIAL RECOMMENDATION ENGINE v4")
    print("=" * 70)

    fibers = load_fiber_database()
    prepregs = load_prepreg_database()
    resins = load_resin_database()

    print(f"\nLoaded {len(fibers)} carbon-fiber grades from the fiber database.")
    print(f"Loaded {len(prepregs)} prepreg systems from the prepreg database.")
    print(f"Loaded {len(resins)} resin systems from the resin database.")

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

    result = build_ranked_results(fibers, req, app)
    print_results(result, req, app, fibers)

    if not result.empty:
        export = ask_yes_no("\nExport complete fiber ranking to CSV? ")
        if export:
            out = BASE_DIR / "recommendation_results_fiber_v4.csv"
            result.to_csv(out, index=False)
            print(f"Saved: {out}")

    # IMPORTANT: These are independent database rankings.
    # No complete fiber+prepreg+resin compatibility is claimed here.
    prepreg_result = build_prepreg_results(prepregs, req, result.head(5))
    resin_result = build_resin_results(resins, req)

    print_system_database_results(prepreg_result, resin_result)

    print(
        "\nNOTE: A prepreg/resin appearing here is a database-level candidate, "
        "not proof of manufacturer-qualified compatibility with the selected fiber."
    )


if __name__ == "__main__":
    main()
