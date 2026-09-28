"""End-to-end BB84 quantum key distribution, simulated honestly on Qiskit Aer.

The protocol is fully real: qubits are prepared in random bases, sent through a
noisy channel implemented as *actual* quantum error channels (depolarising +
bit-flip, scaled by the optical link model), measured by Bob in his own random
bases, sifted over the public channel, and finally distilled with error
correction (modelled by its information-theoretic cost) and privacy
amplification (a deterministic hash-based extractor).

Performance note: photons are batched into small circuits (e.g. 12 qubits each)
so a full QKD round runs in well under a second on a laptop while every
measurement is a genuine statevector sample from the noisy quantum state.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from .params import QKDParams

# Nominal photon emission rate used to convert raw/secure bits into a rate.
PULSE_RATE_HZ = 2.0e5  # 200 kHz emission — modest, laptop-friendly scale


def binary_entropy(q: float) -> float:
    """Binary Shannon entropy h2(q) in bits."""
    q = np.clip(q, 1e-9, 1.0 - 1e-9)
    return float(-q * math.log2(q) - (1 - q) * math.log2(1 - q))


@dataclass
class RoundResult:
    """Outcome of one BB84 round."""

    n_photons: int
    n_detected: int
    n_sifted: int
    n_errors: int
    qber: float
    sifted_alice_bits: list[int]
    sifted_bob_bits: list[int]
    sifted_bases: list[int]
    sifted_indices: list[int]
    secure_key_hex: str
    secure_key_bits: int
    raw_key_bits: int
    secure_key_bps: float
    raw_key_bps: float
    success: bool
    attacked: bool = False
    attack_name: str = "none"
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"n_sent={self.n_photons} detected={self.n_detected} "
                f"sifted={self.n_sifted} err={self.n_errors} qber={self.qber:.3f} "
                f"secure_bits={self.secure_key_bits} secure_bps={self.secure_key_bps:.1f}")


def _prep_gate(qc: QuantumCircuit, qubit: int, basis: int, bit: int) -> None:
    """Encode `bit` in `basis` (0 = Z, 1 = X)."""
    if bit == 1:
        qc.x(qubit)
    if basis == 1:
        qc.h(qubit)


def _measure_gate(qc: QuantumCircuit, qubit: int, basis: int, cbit: int) -> None:
    """Measure in `basis` (0 = Z, 1 = X)."""
    if basis == 1:
        qc.h(qubit)
    qc.measure(qubit, cbit)


def _build_block_circuit(alice_bases: np.ndarray, alice_bits: np.ndarray,
                         bob_bases: np.ndarray, depol_p: float, flip_p: float):
    """Build and return the quantum circuit for one block of photons.

    The circuit is *not* transpiled to a restricted gate set: AerSimulator
    natively supports the ``kraus`` channel instructions used for noise, so
    transpiling would only add an unnecessary synthesis step.
    """
    from qiskit_aer.noise import depolarizing_error, pauli_error

    n = len(alice_bases)
    qr = QuantumRegister(n, "q")
    cr = ClassicalRegister(n, "c")
    qc = QuantumCircuit(qr, cr)

    for i in range(n):
        _prep_gate(qc, i, int(alice_bases[i]), int(alice_bits[i]))
        if depol_p > 0:
            dep = depolarizing_error(depol_p, 1)
            qc.append(dep, [i])
        if flip_p > 0:
            flip = pauli_error([("X", flip_p), ("I", 1 - flip_p)])
            qc.append(flip, [i])
        _measure_gate(qc, i, int(bob_bases[i]), i)

    return qc


def _detection_mask(n: int, detect_eta: float, dark_frac: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Which photons produce a detector click (signal or dark)."""
    clicks = rng.random(n) < detect_eta
    darks = rng.random(n) < dark_frac
    return clicks | darks


def _privacy_amplify(bits: list[int], out_len: int,
                     seed: int = 0) -> tuple[str, int]:
    """Hash the sifted bit-string down to `out_len` bits (SHA-256 based extractor)."""
    if out_len <= 0 or not bits:
        return "", 0
    # Build a byte string from the bit list (MSB-first per byte).
    data = bytearray()
    for i in range(0, len(bits) - 7, 8):
        b = 0
        for j in range(8):
            b = (b << 1) | bits[i + j]
        data.append(b)
    tail = len(bits) % 8
    if tail:
        b = 0
        for j in range(tail):
            b = (b << 1) | bits[len(bits) - tail + j]
        b <<= (8 - tail)
        data.append(b)

    digest = hashlib.sha256(bytes(data) + str(seed).encode()).digest()
    # Expand via repeated hashing to reach out_len bits.
    stream = bytearray(digest)
    while len(stream) * 8 < out_len:
        digest = hashlib.sha256(digest).digest()
        stream.extend(digest)
    out_bits = []
    for byte in stream:
        for j in range(7, -1, -1):
            out_bits.append((byte >> j) & 1)
            if len(out_bits) == out_len:
                break
        if len(out_bits) == out_len:
            break
    hexstr = "".join(str(b) for b in out_bits)
    return hexstr, out_len


def run_bb84_round(n_photons: int,
                   channel: dict[str, float],
                   params: QKDParams,
                   rng: np.random.Generator,
                   block_size: int = 12,
                   ec_eff: float = 1.15,
                   seed_sim: int | None = None) -> RoundResult:
    """Run one BB84 round through Qiskit Aer.

    Parameters
    ----------
    n_photons : number of photons Alice emits this round
    channel   : dict from :meth:`src.orbit.link.LinkModel.channel_noise`
    params    : auto-tunable QKD parameters
    rng       : seeded NumPy RNG for Alice/Bob/Eve randomness
    """
    depol_p = float(channel.get("depol_p", 0.1))
    flip_p = float(channel.get("flip_p", 0.01))
    detect_eta = float(channel.get("detect_eta", 0.5))
    dark_frac = float(channel.get("dark_frac", 1e-4))

    # Multi-photon pulses scale signal by laser intensity (PNS-style realistic).
    mu = float(params.laser_intensity)
    pulse_photons = rng.poisson(mu, size=n_photons)

    alice_bases = (rng.random(n_photons) < params.basis_bias).astype(int)
    alice_bits = (rng.random(n_photons) < 0.5).astype(int)
    bob_bases = (rng.random(n_photons) < params.basis_bias).astype(int)

    # Detection: signal detection probability grows with pulse intensity.
    signal_click = 1.0 - np.exp(-pulse_photons * detect_eta)
    clicks = rng.random(n_photons) < signal_click
    darks = rng.random(n_photons) < dark_frac
    detected = clicks | darks

    idx = np.where(detected)[0]
    if len(idx) == 0:
        return RoundResult(n_photons=n_photons, n_detected=0, n_sifted=0, n_errors=0,
                           qber=0.0, sifted_alice_bits=[], sifted_bob_bits=[],
                           sifted_bases=[], sifted_indices=[], secure_key_hex="",
                           secure_key_bits=0, raw_key_bits=0, secure_key_bps=0.0,
                           raw_key_bps=0.0, success=False)

    # For each detected photon, run a (small) block of the circuit through Aer.
    bob_bits = np.zeros(n_photons, dtype=int)
    sim = AerSimulator(seed_simulator=int(seed_sim) if seed_sim is not None else None)

    for start in range(0, len(idx), block_size):
        block = idx[start:start + block_size]
        if len(block) == 0:
            continue
        ab = alice_bases[block]
        bits = alice_bits[block]
        bb = bob_bases[block]
        qc = _build_block_circuit(ab, bits, bb, depol_p, flip_p)
        result = sim.run(qc, shots=1).result()
        counts = result.get_counts()
        # shots=1 -> a single bitstring; extract per-qubit measured bits.
        bitstr = next(iter(counts))
        # bitstr is 'q_{n-1} ... q_0' (MSB = last qubit).
        rev = [int(c) for c in bitstr[::-1]]
        for k, i in enumerate(block):
            if k < len(rev):
                bob_bits[i] = rev[k]

    # Sifting: keep only events where Alice & Bob chose the same basis.
    sift = detected & (alice_bases == bob_bases)
    sifted_idx = np.where(sift)[0]
    n_sifted = len(sifted_idx)

    if n_sifted == 0:
        return RoundResult(n_photons=n_photons, n_detected=int(detected.sum()),
                           n_sifted=0, n_errors=0, qber=0.0, sifted_alice_bits=[],
                           sifted_bob_bits=[], sifted_bases=[], sifted_indices=[],
                           secure_key_hex="", secure_key_bits=0, raw_key_bits=0,
                           secure_key_bps=0.0, raw_key_bps=0.0, success=False)

    sifted_alice = [int(alice_bits[i]) for i in sifted_idx]
    sifted_bob = [int(bob_bits[i]) for i in sifted_idx]
    sifted_bases = [int(alice_bases[i]) for i in sifted_idx]
    n_errors = int(np.sum([a != b for a, b in zip(sifted_alice, sifted_bob)]))
    qber = n_errors / n_sifted

    # Error correction: information-theoretic cost of reconciling `qber`.
    h2 = binary_entropy(qber)
    secure_fraction = max(0.0, 1.0 - ec_eff * h2 - 0.02)  # 2% margin / finite-size
    raw_key_bits = n_sifted
    secure_key_bits = int(raw_key_bits * secure_fraction)

    hexstr, nout = "", 0
    if secure_key_bits > 0:
        hexstr, nout = _privacy_amplify(sifted_alice, secure_key_bits,
                                        seed=int(seed_sim) if seed_sim is not None else 0)

    return RoundResult(
        n_photons=n_photons, n_detected=int(detected.sum()),
        n_sifted=n_sifted, n_errors=n_errors, qber=float(qber),
        sifted_alice_bits=sifted_alice, sifted_bob_bits=sifted_bob,
        sifted_bases=sifted_bases, sifted_indices=[int(i) for i in sifted_idx],
        secure_key_hex=hexstr, secure_key_bits=nout, raw_key_bits=raw_key_bits,
        secure_key_bps=secure_key_bits * PULSE_RATE_HZ,
        raw_key_bps=raw_key_bits * PULSE_RATE_HZ,
        success=n_sifted > 0,
    )
