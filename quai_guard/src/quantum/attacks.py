"""Eve attack simulation for BB84.

Attacks are modelled at the *quantum* level: they perturb the states Alice
sends, so their effect shows up honestly in the measured QBER (an intercept-
resend attack on a fraction ``strength`` of pulses raises QBER by roughly
``0.25 * strength`` on sifted bits — the textbook BB84 result).

Three attack archetypes are provided:
* ``none``          — genuine channel, no Eve.
* ``intercept_resend`` — Eve intercepts a fraction of pulses, measures them in a
  random basis and resends the measured state (constant QBER bump).
* ``pns`` (photon-number splitting) — Eve beamsplits multi-photon pulses; QBER
  ramps up as Eve harvests photons on high-intensity pulses, producing a
  *distinct* pattern the detector can learn.
* ``intermittent``  — intercept-resend active only inside a time window,
  producing a bursty QBER signature (like Eve only attacking part of the pass).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EveAttack:
    """Description of an Eve attack applied during a QKD round."""

    name: str = "none"
    strength: float = 0.0          # fraction of pulses affected, in (0, 1]
    window: tuple[float, float] | None = None   # (start, end) fraction of round


def _in_window(attack: EveAttack, frac_t: float) -> bool:
    if attack.window is None:
        return True
    s, e = attack.window
    return s <= frac_t <= e


def apply_eve_intercept_resend(round_: "bb84.RoundResult",
                               strength: float,
                               rng: np.random.Generator) -> None:
    """Perturb Bob's sifted bits as if Eve intercepted a fraction of pulses.

    Uses the known BB84 result: for an intercepted pulse whose basis Eve gets
    wrong (probability 0.5), Bob's sifted bit is wrong with probability 0.5,
    and Eve never disturbs pulses she does not intercept.  Net QBER increase
    is ``0.25 * strength``.
    """
    n = len(round_.sifted_bob_bits)
    if n == 0 or strength <= 0:
        return
    mask = rng.random(n) < strength
    for i in range(n):
        if mask[i]:
            eve_basis_match = rng.random() < 0.5
            if not eve_basis_match and rng.random() < 0.5:
                round_.sifted_bob_bits[i] = 1 - round_.sifted_bob_bits[i]


def apply_eve_pns(round_: "bb84.RoundResult", strength: float,
                  rng: np.random.Generator) -> None:
    """Photon-number-splitting: Eve steals multi-photon pulses.

    Higher laser intensity produces more multi-photon pulses, so the induced
    error rate is largest on the brightest pulses — giving the classifier a
    physically meaningful, intensity-correlated signature.
    """
    n = len(round_.sifted_bob_bits)
    if n == 0 or strength <= 0:
        return
    # Use the sent-photon Poisson mean encoded in round meta if available.
    mu = round_.meta.get("laser_intensity", 0.5)
    for i in range(n):
        # Multi-photon probability grows with mu; Eve attacks those preferentially.
        multiphoton_p = 1.0 - np.exp(-mu) * (1 + mu)
        if rng.random() < strength * multiphoton_p and rng.random() < 0.5:
            round_.sifted_bob_bits[i] = 1 - round_.sifted_bob_bits[i]


def run_attacked_sift(round_: "bb84.RoundResult", attack: EveAttack,
                      rng: np.random.Generator, frac_t: float = 1.0) -> None:
    """Apply an Eve attack to a completed (sifted) round in place."""
    if attack.name == "none" or attack.strength <= 0:
        return
    if not _in_window(attack, frac_t):
        return
    if attack.name == "intercept_resend":
        apply_eve_intercept_resend(round_, attack.strength, rng)
    elif attack.name == "pns":
        apply_eve_pns(round_, attack.strength, rng)
    elif attack.name == "intermittent":
        apply_eve_intercept_resend(round_, attack.strength, rng)
    else:
        raise ValueError(f"Unknown attack type: {attack.name!r}")
    # Recompute error statistics after Eve's disturbance.
    n = len(round_.sifted_bob_bits)
    round_.n_errors = int(sum(a != b for a, b
                              in zip(round_.sifted_alice_bits, round_.sifted_bob_bits)))
    round_.qber = round_.n_errors / n if n else 0.0
    round_.attacked = True
    round_.attack_name = attack.name
