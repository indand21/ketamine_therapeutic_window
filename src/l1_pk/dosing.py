"""L0 dosing driver u_e(t) for the L1 engine (TechSpec §3.2, L0 §3.1).

Provides the per-enantiomer mass input rate u_e(t) [mg/h] delivered into the
central compartment. Boluses are represented as narrow finite-rate pulses so
the input remains a well-defined right-hand-side term for solve_ivp; infusions
are piecewise-constant segments.

Regimen conventions (TechSpec §3.2):
  - racemate sets u_S = u_R (split of total mass by R/S fraction);
  - S-ketamine regimen sets u_R = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.l1_pk.config import ENANTIOMERS

# Width [h] of the finite-rate pulse approximating an i.v. bolus. Chosen small
# relative to PK timescales while keeping the RHS smooth for the solver.
_BOLUS_WIDTH_H = 1.0 / 60.0  # 1 minute


@dataclass(frozen=True)
class DoseEvent:
    """Instantaneous i.v. bolus of ``amount`` [mg] at ``time`` [h]."""

    time: float          # h
    amount: float        # mg (total, before enantiomer split)


@dataclass(frozen=True)
class InfusionSegment:
    """Constant-rate infusion of ``rate`` [mg/h] over [start, end) [h]."""

    start: float         # h
    end: float           # h
    rate: float          # mg/h (total, before enantiomer split)


@dataclass(frozen=True)
class DosingRegimen:
    """A complete L0 regimen for one subject.

    ``s_fraction`` is the fraction of total dosed mass that is S-ketamine
    (racemate -> 0.5; pure S-ketamine -> 1.0; R/S ratio sets intermediate
    values). The complementary fraction is R-ketamine.
    """

    boluses: list[DoseEvent] = field(default_factory=list)
    infusions: list[InfusionSegment] = field(default_factory=list)
    s_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.s_fraction <= 1.0:
            raise ValueError("s_fraction must be in [0, 1].")

    def _total_rate(self, t: float) -> float:
        """Total (S+R) input rate [mg/h] at time ``t`` [h]."""
        rate = 0.0
        for seg in self.infusions:
            if seg.start <= t < seg.end:
                rate += seg.rate
        for ev in self.boluses:
            if ev.time <= t < ev.time + _BOLUS_WIDTH_H:
                rate += ev.amount / _BOLUS_WIDTH_H
        return rate

    def input_rate(self, t: float, enantiomer: str) -> float:
        """Return u_e(t) [mg/h] for the given enantiomer (TechSpec §3.2)."""
        if enantiomer not in ENANTIOMERS:
            raise ValueError(f"Unknown enantiomer {enantiomer!r}.")
        frac = self.s_fraction if enantiomer == "S" else (1.0 - self.s_fraction)
        return frac * self._total_rate(t)

    @property
    def bolus_width(self) -> float:
        """Width [h] of the finite-rate pulse used to represent a bolus."""
        return _BOLUS_WIDTH_H

    def has_boluses(self) -> bool:
        """True if the regimen contains at least one bolus event."""
        return len(self.boluses) > 0

    def total_dosed_mass(self, t_end: float) -> float:
        """Total drug mass [mg] delivered up to ``t_end`` [h].

        Used by the mass-conservation verification to form the expected
        cumulative input (TechSpec §7).
        """
        mass = 0.0
        for ev in self.boluses:
            if ev.time < t_end:
                # Pulse fully delivered once t_end passes its window.
                end = min(t_end, ev.time + _BOLUS_WIDTH_H)
                mass += ev.amount * (end - ev.time) / _BOLUS_WIDTH_H
        for seg in self.infusions:
            overlap = max(0.0, min(t_end, seg.end) - seg.start)
            mass += seg.rate * overlap
        return mass
