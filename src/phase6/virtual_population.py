"""Virtual population generation for the QSP framework.

Generates virtual subjects with realistic inter-individual variability
in PK, PD, and injury parameters (Task #23).

Variability sources:
    - CYP2B6 genotype (Indian population frequencies)
    - Injury severity (TBI/aSAH)
    - Age-linked GABAergic reserve
    - SD susceptibility
    - Body weight
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# CYP2B6 allele frequencies (Indian population)
# Li et al. 2015: *6 frequency ~38% in South Asians
CYP2B6_ALLELE_FREQ = {
    "*1": 0.62,  # wild-type
    "*6": 0.38,  # 516G>T + 785A>G (reduced function)
}

# CYP2B6 genotype → CL_met_NK scaling factor
CYP2B6_CL_SCALE = {
    "*1/*1": 1.00,    # wild-type
    "*1/*6": 0.60,    # Li 2015: CL 40.6/68.1
    "*6/*6": 0.32,    # Li 2015: CL 21.6/68.1
}


@dataclass
class VirtualSubject:
    """A single virtual subject with PK/PD variability."""

    subject_id: int

    # Body weight [kg]
    weight: float = 70.0

    # CYP2B6 genotype
    cyp2b6_genotype: str = "*1/*1"
    cyp2b6_cl_scale: float = 1.0

    # Injury severity (0 = healthy, 1 = severe)
    injury_severity: float = 0.0

    # Age [years]
    age: float = 50.0

    # GABAergic reserve (age-linked, 0 = depleted, 1 = full)
    gaba_reserve: float = 1.0

    # SD susceptibility (0 = resistant, 1 = highly susceptible)
    sd_susceptibility: float = 0.5

    # NRHypo sensitivity (0 = resistant, 1 = highly sensitive)
    nrhypo_sensitivity: float = 0.5

    # PK variability (multiplicative)
    pk_cl_variability: float = 1.0  # CL scaling
    pk_vd_variability: float = 1.0  # Vd scaling


@dataclass
class VirtualPopulation:
    """A collection of virtual subjects."""

    subjects: list[VirtualSubject]
    n_subjects: int = 0
    seed: int = 42

    def __post_init__(self):
        self.n_subjects = len(self.subjects)


def generate_virtual_population(
    n_subjects: int = 1000,
    seed: int = 42,
    age_range: tuple[float, float] = (18, 80),
    injury_range: tuple[float, float] = (0.0, 1.0),
) -> VirtualPopulation:
    """Generate a virtual population with realistic variability.

    Args:
        n_subjects: number of virtual subjects
        seed: random seed for reproducibility
        age_range: (min, max) age [years]
        injury_range: (min, max) injury severity

    Returns:
        VirtualPopulation with n_subjects.
    """
    rng = np.random.default_rng(seed)
    subjects = []

    for i in range(n_subjects):
        # Body weight: 70 ± 15 kg (truncated normal)
        weight = np.clip(rng.normal(70, 15), 40, 120)

        # CYP2B6 genotype
        cyp2b6 = _sample_cyp2b6(rng)

        # Age
        age = rng.uniform(*age_range)

        # GABAergic reserve: declines with age
        # Young: ~1.0, elderly: ~0.6 (linked to PV+ interneuron density)
        gaba_reserve = np.clip(1.0 - 0.005 * (age - 20) + rng.normal(0, 0.1), 0.3, 1.0)

        # Injury severity
        injury = rng.uniform(*injury_range)

        # SD susceptibility: increases with injury
        # Healthy: low, injured: high (with inter-individual variability)
        sd_susc = np.clip(0.2 + 0.6 * injury + rng.normal(0, 0.15), 0.0, 1.0)

        # NRHypo sensitivity: increases with injury and age
        nrhypo_sens = np.clip(
            0.3 + 0.3 * injury + 0.2 * (age / 80) + rng.normal(0, 0.1),
            0.0, 1.0,
        )

        # PK variability: log-normal
        pk_cl = rng.lognormal(0, 0.3)  # CV ~30%
        pk_vd = rng.lognormal(0, 0.2)  # CV ~20%

        subjects.append(VirtualSubject(
            subject_id=i,
            weight=weight,
            cyp2b6_genotype=cyp2b6["genotype"],
            cyp2b6_cl_scale=cyp2b6["cl_scale"],
            injury_severity=injury,
            age=age,
            gaba_reserve=gaba_reserve,
            sd_susceptibility=sd_susc,
            nrhypo_sensitivity=nrhypo_sens,
            pk_cl_variability=pk_cl,
            pk_vd_variability=pk_vd,
        ))

    return VirtualPopulation(subjects=subjects, n_subjects=len(subjects), seed=seed)


def _sample_cyp2b6(rng: np.random.Generator) -> dict:
    """Sample a CYP2B6 genotype from Indian population frequencies.

    Returns:
        Dict with 'genotype' and 'cl_scale'.
    """
    # Sample two alleles
    alleles = rng.choice(
        list(CYP2B6_ALLELE_FREQ.keys()),
        size=2,
        p=list(CYP2B6_ALLELE_FREQ.values()),
    )
    # Sort to canonical form
    genotype = "/".join(sorted(alleles))
    # Handle homozygous
    if genotype == "*1/*1":
        pass
    elif genotype == "*1/*6":
        pass
    elif genotype == "*6/*6":
        pass
    else:
        # Fallback
        genotype = "*1/*1"

    return {
        "genotype": genotype,
        "cl_scale": CYP2B6_CL_SCALE.get(genotype, 1.0),
    }


def compute_population_window(
    population: VirtualPopulation,
    dose_range: np.ndarray | None = None,
) -> dict:
    """Compute the therapeutic window across a virtual population.

    For each subject and dose, compute the full QSP stack and determine
    the optimal dose and window boundaries.

    Args:
        population: virtual population
        dose_range: doses to test [mg/kg]

    Returns:
        Dict with population-level window statistics.
    """
    if dose_range is None:
        dose_range = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    # Placeholder: full population simulation would be very expensive
    # This returns the population structure for downstream use
    return {
        "n_subjects": population.n_subjects,
        "dose_range": dose_range,
        "subjects": population.subjects,
    }
