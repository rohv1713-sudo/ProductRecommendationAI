"""
Composite Material Recommendation Engine v5

Evolution from v4: the three independent databases (Fiber, Prepreg, Resin)
are now connected through a Compatibility evidence layer.

Architecture
------------
USER REQUIREMENTS
       ↓
  FIBER FILTER + RANK (engineering hard-constraints → scored candidates)
       ↓
  PREPREG CANDIDATES (process / matrix filter)
       ↓
  RESIN CANDIDATES   (process / matrix filter)
       ↓
  COMPATIBILITY DATABASE
  (fiber_id ↔ prepreg_id evidence: level 0-4)
       ↓
  COMPLETE MATERIAL SYSTEMS
  (fiber + prepreg + compatibility evidence)
       ↓
  RANKED OUTPUT

Compatibility levels
--------------------
0  Unknown
1  Possible / inferred
2  Compatible (documented)
3  Manufacturer-qualified
4  Manufacturer-qualified + laminate/process data

Rules
-----
- Unknown ≠ Incompatible.  A fiber/prepreg pair with no entry stays in the
  ranking but is clearly flagged as UNKNOWN.
- Level ≥ 3 acts as a ranking bonus (not a gate) so documented systems
  surface near the top.
- Hard engineering constraints (strength, modulus, density, elongation) are
  always enforced before compatibility is checked.

Database files expected in the same folder as this script:
    Carbon_Fiber_Master_v6_Top5(1).csv
    Prepreg_Master_Database_v2_Expanded.csv
    Resin_Master_Database_v10.csv
    Compatibility_Master_v1.csv

This engine recommends candidate MATERIAL SYSTEM INPUTS.
It does not replace engineering qualification or physical testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

FIBER_FILE        = BASE_DIR / "Carbon_Fiber_Master_v6_Top5(1).csv"
PREPREG_FILE      = BASE_DIR / "Prepreg_Master_Database_v2_Expanded.csv"
RESIN_FILE        = BASE_DIR / "Resin_Master_Database_v10.csv"
COMPATIBILITY_FILE = BASE_DIR / "Compatibility_Master_v1.csv"


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION PROFILES
# ─────────────────────────────────────────────────────────────────────────────

APPLICATIONS: Dict[str, dict] = {
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

# Manufacturing process options shown to the user.
# These are the COMPOSITE manufacturing processes, not fiber-forming processes.
COMPOSITE_PROCESS_OPTIONS: Dict[str, Optional[str]] = {
    "1": None,
    "2": "VARTM / infusion",
    "3": "RTM",
    "4": "autoclave",
    "5": "OOA / VBO",
    "6": "press / compression molding",
    "7": "filament winding",
    "8": "pultrusion",
}

MATRIX_OPTIONS: Dict[str, Optional[str]] = {
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

# Human-readable compatibility level labels
COMPAT_LABELS: Dict[int, str] = {
    0: "Unknown",
    1: "Possible / inferred",
    2: "Compatible",
    3: "Manufacturer-qualified",
    4: "Manufacturer-qualified + laminate/process data",
}


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean(value: Any) -> str:
    """Return a clean string; empty string for NaN."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    """Normalised (lower-case, stripped) string for comparisons."""
    return clean(value).lower()


def number(value: Any) -> float:
    """Extract the first number from a string; NaN if none found."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def parse_tows(value: Any) -> set:
    """Return the set of tow sizes (integers) mentioned in a string."""
    if pd.isna(value):
        return set()
    return {int(x) for x in re.findall(r"(\d+)\s*K", str(value).upper())}


def ratio_score(value: float, target: float, higher_is_better: bool) -> float:
    """Convert a requirement ratio into a bounded 0-100 preference score."""
    v = number(value)
    t = number(target)
    if not np.isfinite(v) or not np.isfinite(t) or t <= 0:
        return np.nan
    ratio = v / t
    if higher_is_better:
        return float(np.clip(ratio, 0.0, 1.5) / 1.5 * 100.0)
    if v <= t:
        return 100.0
    return float(np.clip(t / v, 0.0, 1.0) * 100.0)


def weighted_mean(values: Dict[str, float], weights: Dict[str, float]) -> float:
    usable = {k: v for k, v in values.items() if np.isfinite(v)}
    if not usable:
        return np.nan
    total_w = sum(weights[k] for k in usable)
    if total_w <= 0:
        return np.nan
    return sum(usable[k] * weights[k] for k in usable) / total_w


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(path: Path, description: str) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(
        f"{description} not found.\n  Expected: {path}"
    )


def load_databases() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and normalise all four databases.

    Returns
    -------
    fibers, prepregs, resins, compatibility
    """
    fibers        = pd.read_csv(_resolve(FIBER_FILE,        "Carbon-fiber database"))
    prepregs      = pd.read_csv(_resolve(PREPREG_FILE,      "Prepreg database"))
    resins        = pd.read_csv(_resolve(RESIN_FILE,        "Resin database"))
    compatibility = pd.read_csv(_resolve(COMPATIBILITY_FILE,"Compatibility database"))

    # ── Fiber numeric columns ──────────────────────────────────────────────
    for col in [
        "tensile_strength_mpa", "tensile_modulus_gpa",
        "ultimate_elongation_percent", "fiber_density_g_cm3",
        "filament_diameter_um", "carbon_content_percent",
    ]:
        if col in fibers.columns:
            fibers[col] = fibers[col].apply(number)

    # ── Prepreg numeric columns ────────────────────────────────────────────
    for col in [
        "fiber_areal_weight_gsm", "resin_content_percent",
        "nominal_cured_ply_thickness_mm", "nominal_fiber_volume_percent",
        "nominal_laminate_density_g_cm3",
        "cure_temperature_C", "Tg_C", "service_temperature_C",
    ]:
        if col in prepregs.columns:
            prepregs[col] = prepregs[col].apply(number)

    # ── Resin numeric columns ──────────────────────────────────────────────
    for col in [
        "density_g_cm3", "viscosity_value", "viscosity_temperature_C",
        "cure_temperature_C", "Tg_C", "service_temperature_C",
    ]:
        if col in resins.columns:
            resins[col] = resins[col].apply(number)

    # ── Compatibility: level must be numeric ───────────────────────────────
    if "compatibility_level" in compatibility.columns:
        compatibility["compatibility_level"] = (
            compatibility["compatibility_level"].apply(number)
        )
    else:
        compatibility["compatibility_level"] = 0.0

    # Normalise the process_id column so matching is case-insensitive.
    if "process_id" in compatibility.columns:
        compatibility["_process_norm"] = (
            compatibility["process_id"].fillna("").astype(str).map(norm)
        )
    else:
        compatibility["_process_norm"] = ""

    return fibers, prepregs, resins, compatibility


# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENTS DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Requirements:
    strength_mpa:       float
    modulus_gpa:        float
    max_density_g_cm3:  float
    elongation_percent: float
    hard_strength:      bool
    hard_modulus:       bool
    hard_density:       bool
    hard_elongation:    bool
    process:            Optional[str]
    matrix:             Optional[str]
    require_certification: bool


# ─────────────────────────────────────────────────────────────────────────────
# HARD-CONSTRAINT FILTER (fiber)
# ─────────────────────────────────────────────────────────────────────────────

def numerical_filter(df: pd.DataFrame, req: Requirements) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    constraints = [
        ("tensile_strength_mpa",      req.strength_mpa,      req.hard_strength,   True),
        ("tensile_modulus_gpa",        req.modulus_gpa,       req.hard_modulus,    True),
        ("ultimate_elongation_percent",req.elongation_percent,req.hard_elongation, True),
        ("fiber_density_g_cm3",        req.max_density_g_cm3, req.hard_density,    False),
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


# ─────────────────────────────────────────────────────────────────────────────
# FIBER SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def engineering_score(row: pd.Series, req: Requirements, app: dict) -> float:
    values = {
        "strength":   ratio_score(row["tensile_strength_mpa"],      req.strength_mpa,      True),
        "modulus":    ratio_score(row["tensile_modulus_gpa"],        req.modulus_gpa,       True),
        "elongation": ratio_score(row["ultimate_elongation_percent"],req.elongation_percent,True),
        "density":    ratio_score(row["fiber_density_g_cm3"],        req.max_density_g_cm3, False),
    }
    importance = app["importance"]
    total = sum(importance.values())
    weights = {k: v / total for k, v in importance.items()}
    return weighted_mean(values, weights)


def application_fit(row: pd.Series, app: dict) -> float:
    terms = app["use_terms"]
    if not terms:
        return 100.0
    text = " ".join(
        norm(row.get(col, ""))
        for col in ["recommended_uses", "market_or_grade", "fiber_class"]
    )
    hits = sum(term in text for term in terms)
    return 100.0 if hits >= 2 else (75.0 if hits == 1 else 50.0)


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


def fiber_process_fit(row: pd.Series, process: Optional[str]) -> float:
    """
    Score a fiber's compatibility with the requested COMPOSITE manufacturing
    process based on its sizing / tow options.
    """
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
    compatibility_text = norm(row.get("sizing_compatibility", ""))
    sizing = norm(row.get("sizing_options", ""))
    if matrix == "thermoplastic":
        return 100.0 if ("unsized" in sizing or "thermoplastic" in compatibility_text) else 45.0
    if matrix in compatibility_text:
        return 100.0
    if matrix == "epoxy" and any(s in sizing for s in ["gp", "gs", " g", "g"]):
        return 95.0
    return 60.0


def certification_fit(row: pd.Series, required: bool) -> float:
    if not required:
        return 100.0
    cert = norm(row.get("certification", ""))
    return 100.0 if cert and cert not in {"not stated", "nan"} else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PREPREG / RESIN SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_PROCESS_ALIASES: Dict[str, List[str]] = {
    "VARTM / infusion": ["vartm", "vacuum infusion", "infusion", "lri"],
    "RTM":              ["rtm", "resin transfer molding"],
    "autoclave":        ["autoclave"],
    "OOA / VBO":        ["ooa", "out-of-autoclave", "vbo", "vacuum bag only"],
    "press / compression molding": ["press", "compression molding"],
    "filament winding": ["filament winding"],
    "pultrusion":       ["pultrusion"],
}


def process_database_score(text: Any, process: Optional[str]) -> float:
    """
    100 when the database field explicitly mentions the requested process.
    50  when the field is present but doesn't match (non-explicit, not incompatible).
    """
    if not process:
        return 100.0
    t = norm(text)
    terms = _PROCESS_ALIASES.get(process, [process])
    return 100.0 if any(term in t for term in terms) else 50.0


def matrix_database_score(text: Any, matrix: Optional[str]) -> float:
    """100 when the database field explicitly lists the requested matrix family."""
    if not matrix:
        return 100.0
    t = norm(text)
    aliases: Dict[str, List[str]] = {
        "epoxy":          ["epoxy"],
        "thermoplastic":  ["thermoplastic", "peek", "pps", "pa", "pekk"],
        "phenolic":       ["phenolic"],
        "vinyl ester":    ["vinyl ester"],
        "cyanate ester":  ["cyanate ester", "cyanate"],
        "BMI":            ["bmi", "bismaleimide"],
        "polyimide":      ["polyimide"],
        "polycarbonate":  ["polycarbonate"],
    }
    terms = aliases.get(matrix, [matrix])
    return 100.0 if any(term in t for term in terms) else 50.0


# ─────────────────────────────────────────────────────────────────────────────
# COMPATIBILITY LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_compatibility(
    fiber_row:  pd.Series,
    prepreg_row: pd.Series,
    compatibility: pd.DataFrame,
    process: Optional[str],
) -> dict:
    """
    Find the strongest documented compatibility evidence for this
    fiber + prepreg combination.

    Matching strategy (both must satisfy):
      1. fiber_id in compatibility  ==  fiber_id in fiber_row   (preferred)
         OR fiber_id is blank in compatibility (prepreg-only entries)
      2. prepreg_id in compatibility  ==  prepreg_id in prepreg_row

    If a process is requested the compatibility process_id must also match.

    Returns a dict with keys: level, label, status, source, statement, id
    """
    fiber_id  = norm(fiber_row.get("fiber_id",  ""))
    prepreg_id = norm(prepreg_row.get("prepreg_id", ""))

    if compatibility.empty:
        return _unknown_compat(process)

    df = compatibility.copy()

    # ── Step 1: match on prepreg_id ────────────────────────────────────────
    if "prepreg_id" in df.columns and prepreg_id:
        prepreg_norm = df["prepreg_id"].fillna("").astype(str).map(norm)
        df = df.loc[prepreg_norm == prepreg_id]

    if df.empty:
        return _unknown_compat(process)

    # ── Step 2: match on fiber_id (blank in DB means prepreg-only record) ──
    if "fiber_id" in df.columns and fiber_id:
        fiber_norm = df["fiber_id"].fillna("").astype(str).map(norm)
        # Keep rows where fiber_id matches OR where DB fiber_id is blank
        df = df.loc[(fiber_norm == fiber_id) | (fiber_norm == "")]

    if df.empty:
        return _unknown_compat(process)

    # ── Step 3: process filter (if requested) ─────────────────────────────
    if process and "_process_norm" in df.columns:
        process_n = norm(process)
        df = df.loc[
            (df["_process_norm"] == process_n) | (df["_process_norm"] == "")
        ]

    if df.empty:
        return _unknown_compat(process)

    # ── Pick best (highest level) row ─────────────────────────────────────
    best = df.sort_values("compatibility_level", ascending=False).iloc[0]
    raw_level = number(best.get("compatibility_level", 0))
    level = int(raw_level) if np.isfinite(raw_level) else 0

    return {
        "level":     level,
        "label":     COMPAT_LABELS.get(level, "Unknown"),
        "status":    "EXPLICIT" if level >= 3 else ("SUPPORTED" if level >= 1 else "UNKNOWN"),
        "source":    clean(best.get("source_document", "")),
        "source_url":clean(best.get("source_url", "")),
        "statement": clean(best.get("manufacturer_statement", "")),
        "id":        clean(best.get("compatibility_id", "")),
        "process":   clean(best.get("process_id", "")),
        "score":     min(level / 4.0, 1.0) * 100.0,
    }


def _unknown_compat(process: Optional[str]) -> dict:
    return {
        "level":      0,
        "label":      COMPAT_LABELS[0],
        "status":     "UNKNOWN",
        "source":     "",
        "source_url": "",
        "statement":  "",
        "id":         "",
        "process":    process or "",
        "score":      np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINT DIAGNOSTICS  (fiber level, shown when no candidate qualifies)
# ─────────────────────────────────────────────────────────────────────────────

def print_constraint_diagnostics(df: pd.DataFrame, req: Requirements) -> None:
    checks = [
        ("Strength",    "tensile_strength_mpa",       req.strength_mpa,      "≥", req.hard_strength,   "MPa"),
        ("Modulus",     "tensile_modulus_gpa",         req.modulus_gpa,       "≥", req.hard_modulus,    "GPa"),
        ("Density",     "fiber_density_g_cm3",         req.max_density_g_cm3, "≤", req.hard_density,    "g/cm³"),
        ("Elongation",  "ultimate_elongation_percent", req.elongation_percent,"≥", req.hard_elongation, "%"),
    ]

    print("\n" + "=" * 70)
    print("CONSTRAINT DIAGNOSTICS")
    print("=" * 70)

    hard_failures = []
    for name, col, threshold, op, hard, unit in checks:
        series = pd.to_numeric(df[col], errors="coerce")
        if op == "≥":
            passed = (series.notna() & (series >= threshold)).sum()
        else:
            passed = (series.notna() & (series <= threshold)).sum()
        missing = series.isna().sum()
        total   = len(df)
        symbol  = "✓" if passed > 0 else "✗"
        hard_t  = "HARD" if hard else "PREFERRED"
        print(
            f"{symbol} {name:<12} {op} {threshold:.2f} {unit:<7} | "
            f"{passed:>3}/{total} pass | {hard_t}"
        )
        if missing:
            print(f"    Missing data: {missing} candidate(s) — "
                  "missing values only eliminate when HARD.")
        if hard and passed == 0:
            hard_failures.append(name)

    if hard_failures:
        print("\nWHY NO CANDIDATE QUALIFIED:")
        for f in hard_failures:
            print(f"  • {f}: no candidate satisfies this HARD constraint.")
        print("\nSUGGESTION: Relax a HARD constraint or verify the value/unit.")
    else:
        print("\nAll HARD constraints have at least one qualifying candidate.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_system_results(
    fibers:        pd.DataFrame,
    prepregs:      pd.DataFrame,
    resins:        pd.DataFrame,
    compatibility: pd.DataFrame,
    req:           Requirements,
    app:           dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build ranked complete material systems.

    Returns
    -------
    systems_df  — one row per (fiber, prepreg) compatibility-linked system
    fibers_df   — ranked fiber-only results (fallback / supplementary)
    """

    # ── 1. Fiber hard-constraint filter ───────────────────────────────────
    fiber_candidates = numerical_filter(fibers, req)

    if req.require_certification and "certification" in fiber_candidates.columns:
        cert_mask = (
            fiber_candidates["certification"]
            .fillna("").astype(str).str.strip().ne("")
        )
        fiber_candidates = fiber_candidates.loc[cert_mask].copy()

    # ── 2. Prepreg candidates (process + matrix soft filter) ──────────────
    prepreg_candidates = prepregs.copy()
    if req.process:
        proc_text = prepreg_candidates["process_suitability"].fillna("").astype(str)
        pass_mask = proc_text.apply(lambda x: process_database_score(x, req.process) == 100.0)
        known_mask = proc_text.str.strip().ne("")
        prepreg_candidates = prepreg_candidates.loc[(~known_mask) | pass_mask].copy()

    if req.matrix:
        matrix_text = prepreg_candidates["matrix_family"].fillna("").astype(str)
        pass_mask = matrix_text.apply(lambda x: matrix_database_score(x, req.matrix) == 100.0)
        known_mask = matrix_text.str.strip().ne("")
        prepreg_candidates = prepreg_candidates.loc[(~known_mask) | pass_mask].copy()

    # ── 3. Score fibers standalone (always computed) ──────────────────────
    fiber_records = []
    for _, fiber in fiber_candidates.iterrows():
        eng        = engineering_score(fiber, req, app)
        app_score  = application_fit(fiber, app)
        grade      = grade_fit(fiber, app["name"])
        proc_fit   = fiber_process_fit(fiber, req.process)
        matrix_fit = fiber_matrix_fit(fiber, req.matrix)
        cert_fit   = certification_fit(fiber, req.require_certification)

        final = (
            0.70 * eng +
            0.08 * app_score +
            0.06 * grade +
            0.08 * proc_fit +
            0.06 * matrix_fit +
            0.02 * cert_fit
        )

        fiber_records.append({
            "Fiber_ID":          fiber.get("fiber_id", ""),
            "Manufacturer":      fiber.get("manufacturer", ""),
            "Product":           fiber.get("product", ""),
            "Strength_MPa":      fiber.get("tensile_strength_mpa", np.nan),
            "Modulus_GPa":       fiber.get("tensile_modulus_gpa", np.nan),
            "Density_g_cm3":     fiber.get("fiber_density_g_cm3", np.nan),
            "Elongation_percent":fiber.get("ultimate_elongation_percent", np.nan),
            "Fiber_Class":       fiber.get("fiber_class", ""),
            "Tow_or_Sizing":     fiber.get("sizing_options", ""),
            "Certification":     fiber.get("certification", ""),
            "Engineering_Score": round(eng, 2),
            "Application_Fit":   round(app_score, 2),
            "Grade_Fit":         round(grade, 2),
            "Process_Fit":       round(proc_fit, 2),
            "Matrix_Fit":        round(matrix_fit, 2),
            "Certification_Fit": round(cert_fit, 2),
            "Final_Score":       round(final, 2),
            "_fiber_row":        fiber,   # keep reference for system pairing
        })

    fibers_df = pd.DataFrame(fiber_records)
    if not fibers_df.empty:
        fibers_df = fibers_df.sort_values("Final_Score", ascending=False).reset_index(drop=True)
        fibers_df["Rank"] = np.arange(1, len(fibers_df) + 1)
        max_score = fibers_df["Final_Score"].max()
        if np.isfinite(max_score) and max_score > 0:
            fibers_df["Relative_Score_percent"] = (
                100.0 * fibers_df["Final_Score"] / max_score
            ).round(1)

    # ── 4. Build fiber × prepreg systems through compatibility layer ───────
    system_records = []

    for _, frow in fiber_records:
        # frow is the dict; extract the actual Series for lookup
        pass

    # Rebuild — iterate the original fiber candidates properly
    fiber_score_map = {
        r["Fiber_ID"]: r for r in fiber_records
    }

    for _, fiber in fiber_candidates.iterrows():
        fid = clean(fiber.get("fiber_id", ""))
        f_scores = fiber_score_map.get(fid, {})
        eng        = f_scores.get("Engineering_Score", np.nan)
        app_score  = f_scores.get("Application_Fit",   np.nan)
        grade      = f_scores.get("Grade_Fit",          np.nan)
        proc_fit   = f_scores.get("Process_Fit",        np.nan)
        matrix_fit = f_scores.get("Matrix_Fit",         np.nan)
        cert_fit   = f_scores.get("Certification_Fit",  np.nan)

        for _, prepreg in prepreg_candidates.iterrows():
            compat = _lookup_compatibility(fiber, prepreg, compatibility, req.process)

            # Scoring formula:
            # When compatibility evidence exists → give it 14% weight.
            # When unknown → fall back to independent-database weights.
            compat_score = compat["score"]
            if np.isfinite(compat_score):
                final = (
                    0.60 * (eng if np.isfinite(eng) else 0) +
                    0.07 * (app_score if np.isfinite(app_score) else 0) +
                    0.05 * (grade if np.isfinite(grade) else 0) +
                    0.07 * (proc_fit if np.isfinite(proc_fit) else 0) +
                    0.05 * (matrix_fit if np.isfinite(matrix_fit) else 0) +
                    0.02 * (cert_fit if np.isfinite(cert_fit) else 0) +
                    0.14 * compat_score
                )
            else:
                final = (
                    0.70 * (eng if np.isfinite(eng) else 0) +
                    0.08 * (app_score if np.isfinite(app_score) else 0) +
                    0.06 * (grade if np.isfinite(grade) else 0) +
                    0.08 * (proc_fit if np.isfinite(proc_fit) else 0) +
                    0.06 * (matrix_fit if np.isfinite(matrix_fit) else 0) +
                    0.02 * (cert_fit if np.isfinite(cert_fit) else 0)
                )

            system_records.append({
                # ── Fiber
                "Fiber_ID":          fid,
                "Fiber_Manufacturer":clean(fiber.get("manufacturer", "")),
                "Fiber":             clean(fiber.get("product", "")),
                "Strength_MPa":      fiber.get("tensile_strength_mpa", np.nan),
                "Modulus_GPa":       fiber.get("tensile_modulus_gpa", np.nan),
                "Density_g_cm3":     fiber.get("fiber_density_g_cm3", np.nan),
                "Elongation_percent":fiber.get("ultimate_elongation_percent", np.nan),
                "Fiber_Class":       clean(fiber.get("fiber_class", "")),
                "Certification":     clean(fiber.get("certification", "")),
                # ── Prepreg
                "Prepreg_ID":        clean(prepreg.get("prepreg_id", "")),
                "Prepreg_Manufacturer":clean(prepreg.get("manufacturer", "")),
                "Prepreg_System":    clean(prepreg.get("product_system", "")),
                "Fiber_Grade_in_Prepreg": clean(prepreg.get("fiber_grade", "")),
                "Matrix":            clean(prepreg.get("matrix_family", "")),
                "FAW_gsm":           prepreg.get("fiber_areal_weight_gsm", np.nan),
                "Resin_Content_pct": prepreg.get("resin_content_percent", np.nan),
                "Ply_Thickness_mm":  prepreg.get("nominal_cured_ply_thickness_mm", np.nan),
                "Vf_pct":            prepreg.get("nominal_fiber_volume_percent", np.nan),
                "Cure_Temp_C":       prepreg.get("cure_temperature_C", np.nan),
                "Tg_C":              prepreg.get("Tg_C", np.nan),
                "Process_Suitability":clean(prepreg.get("process_suitability", "")),
                "Prepreg_Source":    clean(prepreg.get("source_document", "")),
                "Prepreg_Source_URL":clean(prepreg.get("source_url", "")),
                # ── Process
                "Process":           req.process or "",
                # ── Compatibility
                "Compat_Level":      compat["level"],
                "Compat_Label":      compat["label"],
                "Compat_Status":     compat["status"],
                "Compat_ID":         compat["id"],
                "Compat_Source":     compat["source"],
                "Compat_Source_URL": compat["source_url"],
                "Manufacturer_Statement": compat["statement"],
                # ── Scores
                "Engineering_Score": round(eng, 2)       if np.isfinite(eng)        else np.nan,
                "Application_Fit":   round(app_score, 2) if np.isfinite(app_score)  else np.nan,
                "Grade_Fit":         round(grade, 2)     if np.isfinite(grade)       else np.nan,
                "Process_Fit":       round(proc_fit, 2)  if np.isfinite(proc_fit)   else np.nan,
                "Matrix_Fit":        round(matrix_fit, 2)if np.isfinite(matrix_fit) else np.nan,
                "Certification_Fit": round(cert_fit, 2)  if np.isfinite(cert_fit)   else np.nan,
                "Compat_Score":      round(compat_score, 2) if np.isfinite(compat_score) else np.nan,
                "Final_Score":       round(final, 2),
            })

    if not system_records:
        return pd.DataFrame(), fibers_df

    systems_df = (
        pd.DataFrame(system_records)
        .sort_values(
            ["Compat_Level", "Final_Score"],
            ascending=[False, False]
        )
        .reset_index(drop=True)
    )
    systems_df["Rank"] = np.arange(1, len(systems_df) + 1)

    max_score = pd.to_numeric(systems_df["Final_Score"], errors="coerce").max()
    if np.isfinite(max_score) and max_score > 0:
        systems_df["Relative_Score_percent"] = (
            100.0 * systems_df["Final_Score"] / max_score
        ).round(1)
    else:
        systems_df["Relative_Score_percent"] = np.nan

    return systems_df, fibers_df


# ─────────────────────────────────────────────────────────────────────────────
# PRINT OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _val(v: Any, fmt: str = "", fallback: str = "N/A") -> str:
    """Format a number or return fallback."""
    try:
        f = float(v)
        if not np.isfinite(f):
            return fallback
        return format(f, fmt)
    except (TypeError, ValueError):
        return fallback if not str(v).strip() else str(v).strip()


def print_results(
    systems_df: pd.DataFrame,
    fibers_df:  pd.DataFrame,
    req:        Requirements,
    app:        dict,
    all_fibers: pd.DataFrame,
) -> None:
    sep = "━" * 70

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  RECOMMENDED MATERIAL SYSTEMS")
    print(sep)
    print(
        f"\n  Application : {app['name']}\n"
        f"  Process     : {req.process or 'Not specified'}\n"
        f"  Matrix      : {req.matrix  or 'Not specified'}"
    )

    # ── Case A: no fiber candidates at all ────────────────────────────────
    if fibers_df.empty:
        print("\n  NO CANDIDATE SATISFIES THE HARD FIBER REQUIREMENTS.\n")
        print_constraint_diagnostics(all_fibers, req)
        return

    # ── Case B: fiber candidates exist but no system pairings found ────────
    if systems_df.empty:
        print(
            "\n  No complete material systems were produced.\n"
            "  This happens when the prepreg database has no records matching\n"
            "  the process/matrix filter, or when the Cartesian product is empty.\n"
            "  Showing top fiber recommendations only.\n"
        )
        _print_fiber_only(fibers_df, req)
        return

    # ── Case C: full system results ────────────────────────────────────────
    top = systems_df.head(5)
    for _, r in top.iterrows():
        score_txt = _val(r.get("Relative_Score_percent"), ".1f", "N/A")
        status    = r.get("Compat_Status", "UNKNOWN")
        level     = r.get("Compat_Level", 0)

        print(f"\n{sep}")
        print(
            f"  #{int(r['Rank'])}  {r.get('Fiber_Manufacturer','')} "
            f"{r.get('Fiber','')}  —  {score_txt}%"
        )
        print(sep)

        # Fiber line
        print(
            f"  Fiber       : {r.get('Fiber','')} "
            f"({r.get('Fiber_Manufacturer','')})"
        )
        print(
            f"    Strength  : {_val(r.get('Strength_MPa'), '.0f')} MPa  |  "
            f"Modulus: {_val(r.get('Modulus_GPa'), '.0f')} GPa  |  "
            f"Density: {_val(r.get('Density_g_cm3'), '.2f')} g/cm³  |  "
            f"Elong: {_val(r.get('Elongation_percent'), '.1f')}%"
        )
        print(f"    Class     : {r.get('Fiber_Class','N/A')}")
        cert = r.get("Certification","")
        print(f"    Cert      : {cert if cert else 'Not stated'}")

        # Prepreg line
        print(
            f"\n  Prepreg     : {r.get('Prepreg_System','')} "
            f"({r.get('Prepreg_Manufacturer','')})"
        )
        print(
            f"    ID        : {r.get('Prepreg_ID','')}"
        )
        print(
            f"    Fiber in prepreg : {r.get('Fiber_Grade_in_Prepreg','N/A')}"
        )
        print(
            f"    Matrix    : {r.get('Matrix','N/A')}"
        )
        print(
            f"    FAW       : {_val(r.get('FAW_gsm'), '.0f', 'N/A')} gsm  |  "
            f"Resin: {_val(r.get('Resin_Content_pct'), '.0f', 'N/A')} %  |  "
            f"Vf: {_val(r.get('Vf_pct'), '.0f', 'N/A')} %"
        )
        print(
            f"    Ply thk   : {_val(r.get('Ply_Thickness_mm'), '.3f', 'N/A')} mm  |  "
            f"Cure: {_val(r.get('Cure_Temp_C'), '.0f', 'N/A')} °C  |  "
            f"Tg: {_val(r.get('Tg_C'), '.0f', 'N/A')} °C"
        )
        print(f"    Process   : {r.get('Process_Suitability','N/A')}")
        src = r.get('Prepreg_Source','')
        print(f"    Source    : {src if src else 'Not stated'}")

        # Compatibility block
        print(f"\n  COMPATIBILITY")
        if status == "UNKNOWN":
            print(
                f"    Status    : UNKNOWN\n"
                f"    Level     : 0 — {COMPAT_LABELS[0]}\n"
                f"    Note      : No entry in Compatibility_Master_v1.csv\n"
                f"                Unknown ≠ Incompatible."
            )
        else:
            level_int = int(level) if np.isfinite(float(level)) else 0
            print(f"    Status    : {status}")
            print(f"    Level     : {level_int} — {COMPAT_LABELS.get(level_int, 'Unknown')}")
            print(f"    ID        : {r.get('Compat_ID','')}")
            compat_src = r.get('Compat_Source','')
            print(f"    Source    : {compat_src if compat_src else 'Not stated'}")
            stmt = r.get("Manufacturer_Statement","")
            if stmt:
                # Wrap long statements
                words = stmt.split()
                lines, line = [], []
                for w in words:
                    line.append(w)
                    if len(" ".join(line)) > 60:
                        lines.append("              " + " ".join(line))
                        line = []
                if line:
                    lines.append("              " + " ".join(line))
                print("    Statement :")
                print("\n".join(lines))

        # Score breakdown
        print(f"\n  SCORES")
        print(
            f"    Engineering : {_val(r.get('Engineering_Score'), '.1f')}  |  "
            f"Application : {_val(r.get('Application_Fit'), '.1f')}  |  "
            f"Grade : {_val(r.get('Grade_Fit'), '.1f')}"
        )
        print(
            f"    Process     : {_val(r.get('Process_Fit'), '.1f')}  |  "
            f"Matrix      : {_val(r.get('Matrix_Fit'), '.1f')}  |  "
            f"Compat : {_val(r.get('Compat_Score'), '.1f', 'N/A (UNKNOWN)')}"
        )
        print(
            f"    FINAL SCORE : {_val(r.get('Final_Score'), '.2f')}  "
            f"({score_txt}% of leader)"
        )

    print(f"\n{sep}")

    # Summary note
    print(
        f"\n  Showing top {min(5, len(systems_df))} of {len(systems_df)} "
        f"system combinations.\n"
        f"  Compatibility levels: 0=Unknown | 1=Possible | 2=Compatible | "
        f"3=Qualified | 4=Qualified+data\n"
        f"  UNKNOWN does not mean incompatible — it means no documented "
        f"evidence exists yet in the database.\n"
    )


def _print_fiber_only(fibers_df: pd.DataFrame, req: Requirements) -> None:
    sep = "━" * 70
    print(f"\n{sep}")
    print("  TOP FIBER RECOMMENDATIONS (standalone)")
    print(sep)
    for _, r in fibers_df.head(5).iterrows():
        score = r.get("Relative_Score_percent", np.nan)
        score_txt = f"{float(score):.1f}%" if pd.notna(score) else "N/A"
        print(
            f"\n  #{int(r['Rank'])} {r.get('Manufacturer','')} "
            f"{r.get('Product','')} — {score_txt}"
        )
        print(
            f"    {_val(r.get('Strength_MPa'),'.0f')} MPa | "
            f"{_val(r.get('Modulus_GPa'),'.0f')} GPa | "
            f"{_val(r.get('Density_g_cm3'),'.2f')} g/cm³ | "
            f"{_val(r.get('Elongation_percent'),'.1f')}%"
        )
        print(
            f"    Engineering: {_val(r.get('Engineering_Score'),'.1f')} | "
            f"Process: {_val(r.get('Process_Fit'),'.1f')} | "
            f"Matrix: {_val(r.get('Matrix_Fit'),'.1f')}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# USER INPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ask_float(prompt: str, minimum: float = 0.0) -> float:
    while True:
        try:
            v = float(input(prompt))
            if v < minimum:
                raise ValueError
            return v
        except ValueError:
            print(f"  Enter a number >= {minimum}.")


def ask_yes_no(prompt: str) -> bool:
    while True:
        a = input(prompt).strip().lower()
        if a in {"y", "yes"}:
            return True
        if a in {"n", "no"}:
            return False
        print("  Enter yes or no.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("  COMPOSITE MATERIAL RECOMMENDATION ENGINE  v5")
    print("  Fiber · Prepreg · Compatibility-linked Material Systems")
    print("=" * 70)

    print("\nLoading databases…")
    fibers, prepregs, resins, compatibility = load_databases()

    print(f"  ✓ Fiber database        : {len(fibers)} grades")
    print(f"  ✓ Prepreg database      : {len(prepregs)} records")
    print(f"  ✓ Resin database        : {len(resins)} records")
    print(f"  ✓ Compatibility database: {len(compatibility)} evidence records")

    # ── Application ───────────────────────────────────────────────────────
    print("\nAPPLICATION")
    for key, val in APPLICATIONS.items():
        print(f"  {key}. {val['name']}")

    choice = input("Choose application: ").strip()
    while choice not in APPLICATIONS:
        choice = input("  Choose a valid option (1-6): ").strip()

    app = APPLICATIONS[choice].copy()

    if choice == "6":
        print("\nCUSTOM IMPORTANCE (enter 1-10 for each)")
        app["importance"] = {
            "strength":   ask_float("  Strength importance  : ", 1),
            "modulus":    ask_float("  Modulus importance   : ", 1),
            "density":    ask_float("  Density importance   : ", 1),
            "elongation": ask_float("  Elongation importance: ", 1),
        }

    # ── Engineering requirements ───────────────────────────────────────────
    print("\nENGINEERING REQUIREMENTS")
    req = Requirements(
        strength_mpa       = ask_float("  Minimum tensile strength (MPa) : "),
        modulus_gpa        = ask_float("  Minimum tensile modulus (GPa)  : "),
        max_density_g_cm3  = ask_float("  Maximum density (g/cm³)        : "),
        elongation_percent = ask_float("  Minimum elongation (%)         : "),
        hard_strength      = ask_yes_no("  Strength is a HARD constraint? (y/n): "),
        hard_modulus       = ask_yes_no("  Modulus  is a HARD constraint? (y/n): "),
        hard_density       = ask_yes_no("  Density  is a HARD constraint? (y/n): "),
        hard_elongation    = ask_yes_no("  Elongation is HARD?            (y/n): "),
        process            = None,
        matrix             = None,
        require_certification = False,
    )

    # ── Composite manufacturing process ───────────────────────────────────
    print("\nCOMPOSITE MANUFACTURING PROCESS")
    for key, name in COMPOSITE_PROCESS_OPTIONS.items():
        print(f"  {key}. {name or 'No process preference'}")
    req.process = COMPOSITE_PROCESS_OPTIONS.get(input("  Choose process: ").strip())

    # ── Matrix family ─────────────────────────────────────────────────────
    print("\nMATRIX FAMILY")
    for key, name in MATRIX_OPTIONS.items():
        print(f"  {key}. {name or 'No matrix preference'}")
    req.matrix = MATRIX_OPTIONS.get(input("  Choose matrix: ").strip())

    # ── Certification ──────────────────────────────────────────────────────
    req.require_certification = ask_yes_no(
        "\n  Require an explicitly stated certification? (y/n): "
    )

    # ── Run ───────────────────────────────────────────────────────────────
    print("\nRunning recommendation engine…")
    systems_df, fibers_df = build_system_results(
        fibers, prepregs, resins, compatibility, req, app
    )

    print_results(systems_df, fibers_df, req, app, fibers)

    # ── Export ────────────────────────────────────────────────────────────
    if not systems_df.empty:
        if ask_yes_no("\n  Export complete system ranking to CSV? (y/n): "):
            out = BASE_DIR / "recommendation_results_v5.csv"
            # Drop internal helper column before export
            export_df = systems_df.copy()
            export_df.to_csv(out, index=False)
            print(f"  Saved: {out}")

    elif not fibers_df.empty:
        if ask_yes_no("\n  Export fiber-only ranking to CSV? (y/n): "):
            out = BASE_DIR / "recommendation_results_fiber_v5.csv"
            fibers_df.drop(columns=["_fiber_row"], errors="ignore").to_csv(out, index=False)
            print(f"  Saved: {out}")

    print(
        "\n  NOTE: This output is a database-level recommendation.\n"
        "  It does not replace manufacturer qualification, physical testing,\n"
        "  or engineering analysis.\n"
    )


if __name__ == "__main__":
    main()
