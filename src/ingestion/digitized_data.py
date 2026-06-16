"""Digitized published concentration-time data for L1 calibration.

Concentration-time curves are loaded at import time from the genuinely
figure-digitized CSVs in ``docs/validation/digitized/`` (produced by
``scripts/autodigitize.py`` and ``scripts/kamp_digitize.py`` via reproducible
pixel digitization). This module previously held concentration arrays that were
*reconstructed* from published PK parameters; those have been replaced with the
real digitized data. See ``docs/validation/digitized/README.md`` for the
digitization method, validation self-checks, and caveats.

Each :class:`DigitizedCurve` includes:
  - Source citation
  - Dosing regimen used
  - Species (KET_S, KET_R, NK_S, NK_R, HNK)
  - Compartment (plasma, ecf)
  - Time points [h] and concentrations [mg/L]

All concentrations are converted to **mg/L** (the L1 model output unit) on load:
  - Hasan curves are digitized in ng/mL    → mg/L  (× 1e-3)
  - Kamp curves are digitized in nmol/mL   → mg/L  (× MW_species × 1e-3)

Tabulated summary statistics (Hasan Table 1, Perez-Ruixo, Weiss, Moaddel) are
literature transcriptions, not reconstructions, and are kept verbatim.

Kamp 2020 regimen
-----------------
Kamp 2020 administered *three increasing i.v. doses* (an escalating step
infusion over 180 min) of esketamine and racemic ketamine to 20 healthy
volunteers, so every Fig 1 curve peaks at ~3 h. This regimen is implemented as
:func:`src.validation.l1_pk_validation.kamp_2020_escalating_regimen` and the
calibrator builds it automatically for any curve whose ``regimen`` string
contains "escalating" (see ``_build_regimen_for_curve``). The Kamp Fig 1 curves
are therefore now part of :data:`ALL_DIGITIZED_CURVES` (the default calibration
set), alongside the Hasan IV curves. DHNK panels (g, h, i) are digitized but
DHNK is not an L1 species, so they live only in :data:`KAMP_2020_DHNK_CURVES` /
:data:`ALL_REFERENCE_CURVES`.

Sources:
  Kamp 2020, Br J Anaesth 125(5):750-761 (DOI: 10.1016/j.bja.2020.06.067)
  Hasan 2021, Anesthesiology 135(2):326-339 (DOI: 10.1097/ALN.0000000000003829)
  Perez-Ruixo 2021, Clin Pharmacokinet (DOI: 10.1007/s40262-020-00958-8)
  Weiss & Siegmund 2022, Clin Pharmacol Drug Dev (DOI: 10.1002/cpdd.1181)
  Moaddel 2023, iScience (DOI: 10.1016/j.isci.2023.107856)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class DigitizedCurve:
    """A single digitized concentration-time curve from a published figure."""
    source: str              # e.g., "Kamp 2020, Fig 1a"
    species: str             # "KET_S", "KET_R", "NK_S", "NK_R", "HNK"
    compartment: str         # "plasma", "ecf"
    regimen: str             # Description of dosing
    weight_kg: float         # Subject weight (for dose scaling)
    dose_mg: float           # Dose in mg (for AUC normalization)
    times_h: np.ndarray      # Time points [h]
    concentrations: np.ndarray  # Concentrations [mg/L]
    unit: str = "mg/L"
    notes: str = ""


# ===========================================================================
# CSV loading infrastructure
# ===========================================================================

# Repo-root-relative path resolved from this file → cwd-independent.
_DIGITIZED_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "validation" / "digitized"
)

# Molar masses [g/mol] for nmol/mL → ng/mL conversion (Kamp curves).
_MW = {
    "KET_S": 237.73, "KET_R": 237.73,   # ketamine
    "NK_S": 223.70, "NK_R": 223.70,     # norketamine
    "DHNK_S": 221.68, "DHNK_R": 221.68,  # dehydronorketamine
    "HNK": 239.70,                       # hydroxynorketamine
}

# Kamp Fig 1 panel → (model species, formulation arm). DHNK panels (g, h, i)
# are digitized but DHNK is not a species in the L1 model.
_KAMP_PANELS = {
    "a": ("KET_S", "esketamine"), "b": ("KET_S", "racemic"), "c": ("KET_R", "racemic"),
    "d": ("NK_S", "esketamine"),  "e": ("NK_S", "racemic"),  "f": ("NK_R", "racemic"),
    "g": ("DHNK_S", "esketamine"), "h": ("DHNK_S", "racemic"), "i": ("DHNK_R", "racemic"),
    "j": ("HNK", "racemic"),
}


def _read_csv(path: Path) -> tuple[dict[str, str], np.ndarray, np.ndarray, str]:
    """Parse a digitized CSV.

    Returns (header_meta, times_h, raw_concentrations, conc_unit_token).
    Header lines look like ``# key,value``; the data block starts at the
    ``time_h,<conc_col>`` line, whose second token names the unit.
    """
    meta: dict[str, str] = {}
    times: list[float] = []
    concs: list[float] = []
    conc_unit = ""

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if "," in body:
                k, v = body.split(",", 1)
                meta[k.strip()] = v.strip().strip('"')
            continue
        if line.startswith("time_h"):
            parts = line.split(",")
            if len(parts) > 1:
                conc_unit = parts[1].strip()
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                times.append(float(parts[0]))
                concs.append(float(parts[1]))
            except ValueError:
                continue

    return meta, np.array(times), np.array(concs), conc_unit


def _to_mg_per_L(raw: np.ndarray, unit_token: str, species: str) -> np.ndarray:
    """Convert digitized concentrations to mg/L based on the column unit."""
    u = unit_token.lower()
    if "nmol" in u:
        # nmol/mL × MW(g/mol) = ng/mL;  ng/mL × 1e-3 = mg/L
        mw = _MW.get(species, _MW["KET_S"])
        return raw * mw * 1e-3
    if "ng" in u:
        # ng/mL × 1e-3 = mg/L
        return raw * 1e-3
    # Already mg/L (or unknown) → pass through.
    return raw


def _load_hasan(species: str) -> DigitizedCurve:
    fname = "hasan_S_ketamine_iv5mg.csv" if species == "KET_S" else "hasan_R_ketamine_iv5mg.csv"
    path = _DIGITIZED_DIR / fname
    meta, t, raw, unit = _read_csv(path)
    conc = _to_mg_per_L(raw, unit, species)
    return DigitizedCurve(
        source=meta.get("source", path.stem),
        species=species,
        compartment="plasma",
        regimen="5 mg racemate IV over 30 min",
        weight_kg=70.0,
        dose_mg=2.5,  # half of 5 mg racemate per enantiomer
        times_h=t,
        concentrations=conc,
        notes=(
            f"Digitized from Hasan 2021 Fig 1 ({fname}). Mean-line apex "
            "underestimates published Cmax (15.6/16.5 ng/mL) by ~10-20% — "
            "use the table CSV for Cmax anchors; decay phase is reliable."
        ),
    )


# Kamp 2020 / Olofsen 2021 escalating-infusion rates [mg/kg/h] per 60-min step
# (mirrors src.validation.l1_pk_validation._KAMP_ESCALATING_RATES). Used here
# only to compute a representative cumulative dose_mg at the 70 kg reference.
_KAMP_RATES = {"esketamine": (0.14, 0.28, 0.57), "racemic": (0.28, 0.57, 1.14)}
_KAMP_REF_WEIGHT_KG = 70.0


def _load_kamp(panel: str) -> DigitizedCurve:
    species, arm = _KAMP_PANELS[panel]
    matches = sorted(_DIGITIZED_DIR.glob(f"kamp_{panel}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No digitized CSV for Kamp panel {panel!r}")
    path = matches[0]
    meta, t, raw, unit = _read_csv(path)
    conc = _to_mg_per_L(raw, unit, species)
    # Cumulative dose at the 70 kg reference. Esketamine is pure S; racemic
    # splits 50/50, so each enantiomer (and its metabolites) sees half. Total
    # HNK is fed by both enantiomers → use the full racemic parent dose.
    total_mg = sum(_KAMP_RATES[arm]) * _KAMP_REF_WEIGHT_KG
    if arm == "esketamine":
        dose_mg = total_mg              # all S
    elif species == "HNK":
        dose_mg = total_mg             # total parent (S+R)
    else:
        dose_mg = total_mg / 2.0       # per enantiomer
    return DigitizedCurve(
        source=meta.get("source", path.stem),
        species=species,
        compartment="plasma",
        regimen=f"{arm} escalating IV infusion (Kamp 2020: 3×60-min steps over 180 min)",
        weight_kg=_KAMP_REF_WEIGHT_KG,
        dose_mg=dose_mg,
        times_h=t,
        concentrations=conc,
        notes=(
            f"Digitized from Kamp 2020 Fig 1 panel {panel} ({path.name}). "
            "Escalating 3-step i.v. infusion (Tmax ~3 h); simulate with "
            "src.validation.l1_pk_validation.kamp_2020_escalating_regimen."
        ),
    )


# ===========================================================================
# Loaded curves
# ===========================================================================

# Hasan 2021 — IV 5 mg racemate over 30 min (clean short IV; Tmax ~0.5 h).
HASAN_2021_S_KET_PLASMA = _load_hasan("KET_S")
HASAN_2021_R_KET_PLASMA = _load_hasan("KET_R")

# Kamp 2020 Fig 1 — escalating IV infusion (Tmax ~3 h). Backward-compatible
# named symbols (parent + norketamine + total HNK; DHNK kept separately).
KAMP_2020_S_KET_ESKETAMINE = _load_kamp("a")
KAMP_2020_S_KET_RACEMIC = _load_kamp("b")
KAMP_2020_R_KET_RACEMIC = _load_kamp("c")
KAMP_2020_S_NK_ESKETAMINE = _load_kamp("d")
KAMP_2020_S_NK_RACEMIC = _load_kamp("e")
KAMP_2020_R_NK_RACEMIC = _load_kamp("f")
KAMP_2020_HNK_RACEMIC = _load_kamp("j")

# DHNK is digitized but not a species in the L1 model — kept for reference only.
KAMP_2020_DHNK_CURVES: list[DigitizedCurve] = [
    _load_kamp("g"), _load_kamp("h"), _load_kamp("i"),
]

# All model-relevant Kamp Fig 1 curves (escalating infusion; reference use).
KAMP_2020_FIG1_CURVES: list[DigitizedCurve] = [
    KAMP_2020_S_KET_ESKETAMINE,
    KAMP_2020_S_KET_RACEMIC,
    KAMP_2020_R_KET_RACEMIC,
    KAMP_2020_S_NK_ESKETAMINE,
    KAMP_2020_S_NK_RACEMIC,
    KAMP_2020_R_NK_RACEMIC,
    KAMP_2020_HNK_RACEMIC,
]

HASAN_2021_HNK_AUC_RATIO = {
    "source": "Hasan 2021, Anesthesiology (Results)",
    "regimen": "5 mg racemate IV over 30 min",
    "HNK_KET_AUC_ratio_S": 1.7,   # S-HNK/S-KET AUC ratio (IV)
    "HNK_KET_AUC_ratio_R": 3.1,   # R-HNK/R-KET AUC ratio (IV)
    "HNK_KET_AUC_ratio_S_SD": 0.8,
    "HNK_KET_AUC_ratio_R_SD": 1.4,
    "notes": "Oral ratios much higher: 18x (S), 30x (R) at 40 mg tablet",
}

# ===========================================================================
# Perez-Ruixo et al. 2021, Clin Pharmacokinet — Tabulated summary stats
# ===========================================================================
# N=820 across 12 trials (IN/IV/oral esketamine). Population PK estimates.

PEREZ_RUIXO_2021_SUMMARY = {
    "source": "Perez-Ruixo 2021, Clin Pharmacokinet (N=820, 12 trials)",
    "CL_S_L_per_h": 114.0,         # Total clearance S-KET [L/h]
    "CL_R_L_per_h": 107.6,         # Total clearance R-KET [L/h]
    "Vss_S_L": 752.0,              # Vdss S-KET [L]
    "Vss_R_L": 638.0,              # Vdss R-KET [L]
    "f_m_S": 0.54,                 # Fraction to NK (S) — FRn
    "f_m_R": 0.50,                 # Fraction to NK (R)
    "F_oral": 0.186,               # Oral bioavailability
    "t_half_S_h_range": (4.5, 6.0),  # Terminal half-life range [h]
    "t_half_R_h_range": (5.0, 7.0),  # Terminal half-life range [h]
}

# ===========================================================================
# Hasan 2021 — Complete tabulated PK parameters (from Table 1)
# ===========================================================================

HASAN_2021_SUMMARY = {
    "source": "Hasan 2021, Anesthesiology 135(2):326-339 (N=15, IV 5 mg + oral 10-80 mg)",
    # IV 5 mg racemate over 30 min:
    "IV_Cmax_S_ng_mL": 15.6,       # S-KET Cmax
    "IV_Cmax_S_SD_ng_mL": 4.5,
    "IV_Cmax_R_ng_mL": 16.5,       # R-KET Cmax
    "IV_Cmax_R_SD_ng_mL": 4.9,
    "IV_Vdss_S_L_per_kg": 6.6,     # Vdss S-KET
    "IV_Vdss_S_SD_L_per_kg": 2.2,
    "IV_Vdss_R_L_per_kg": 5.6,     # Vdss R-KET
    "IV_Vdss_R_SD_L_per_kg": 2.1,
    "IV_thalf_S_h": 5.2,           # t½ S-KET
    "IV_thalf_S_SD_h": 3.4,
    "IV_thalf_R_h": 6.1,           # t½ R-KET
    "IV_thalf_R_SD_h": 3.1,
    "IV_CL_S_mL_per_min": 1620,    # CL S-KET
    "IV_CL_S_SD_mL_per_min": 380,
    "IV_CL_R_mL_per_min": 1530,    # CL R-KET
    "IV_CL_R_SD_mL_per_min": 380,
    "IV_HNK_KET_AUC_ratio_S": 1.7,  # AUC HNK/AUC KET (S)
    "IV_HNK_KET_AUC_ratio_S_SD": 0.8,
    "IV_HNK_KET_AUC_ratio_R": 3.1,  # AUC HNK/AUC KET (R)
    "IV_HNK_KET_AUC_ratio_R_SD": 1.4,
    # Oral 40 mg prolonged-release:
    "Oral_F_S_pct": 15,            # Bioavailability S (%)
    "Oral_F_S_SD_pct": 8,
    "Oral_F_R_pct": 19,            # Bioavailability R (%)
    "Oral_F_R_SD_pct": 10,
    "Oral_thalf_S_h": 11,
    "Oral_thalf_S_SD_h": 4,
    "Oral_thalf_R_h": 10,
    "Oral_thalf_R_SD_h": 4,
    "Oral_HNK_KET_AUC_ratio_S": 18,  # Much higher oral vs IV
    "Oral_HNK_KET_AUC_ratio_S_SD": 11,
    "Oral_HNK_KET_AUC_ratio_R": 30,
    "Oral_HNK_KET_AUC_ratio_R_SD": 16,
    "Renal_excretion_S_pct": 7,    # % dose renally excreted (S)
    "Renal_excretion_R_pct": 17,   # % dose renally excreted (R)
}

# ===========================================================================
# Weiss & Siegmund 2022 — HNK:KET steady-state ratios
# ===========================================================================

WEISS_2022_SUMMARY = {
    "source": "Weiss & Siegmund 2022, Clin Pharmacol Drug Dev",
    "HNK_KET_ratio_S_steady_state": 14.0,  # HNK:S-KET at SS
    "HNK_KET_ratio_R_steady_state": 46.0,  # HNK:R-KET at SS
    "HNK_NK_ratio_steady_state": 15.0,     # HNK:NK at SS
}

# ===========================================================================
# Moaddel et al. 2023 — CSF/plasma partitioning
# ===========================================================================

MOADDEL_2023_SUMMARY = {
    "source": "Moaddel 2023, iScience (N=9, paired CSF+plasma)",
    "CSF_plasma_ratio_KET": 0.4,
    "fu_plasma_S": 0.65,
    "fu_plasma_R": 0.71,
    "Kp_uu_estimated": 0.6,  # 0.4 / 0.65
}


# ===========================================================================
# Curve collections
# ===========================================================================

# Calibration set: all curves mapped to an L1 species whose regimen the
# calibrator can build — Hasan IV (Tmax ~0.5 h) and Kamp Fig 1 escalating
# infusion (Tmax ~3 h, via kamp_2020_escalating_regimen).
ALL_DIGITIZED_CURVES: list[DigitizedCurve] = [
    HASAN_2021_S_KET_PLASMA,
    HASAN_2021_R_KET_PLASMA,
] + KAMP_2020_FIG1_CURVES

# Calibration set plus the DHNK curves (DHNK is not an L1 species, so it is not
# calibrated — kept for plotting / visual validation only).
ALL_REFERENCE_CURVES: list[DigitizedCurve] = (
    ALL_DIGITIZED_CURVES + KAMP_2020_DHNK_CURVES
)


def get_curves_for_species(
    species: str, curves: list[DigitizedCurve] | None = None
) -> list[DigitizedCurve]:
    """Return all curves for a given species (defaults to the reference set)."""
    pool = ALL_REFERENCE_CURVES if curves is None else curves
    return [c for c in pool if c.species == species]


def get_curves_for_source(
    source_prefix: str, curves: list[DigitizedCurve] | None = None
) -> list[DigitizedCurve]:
    """Return all curves from a given source (defaults to the reference set)."""
    pool = ALL_REFERENCE_CURVES if curves is None else curves
    return [c for c in pool if c.source.startswith(source_prefix)]
