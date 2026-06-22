"""Through-port microring TMM model.

The current model is for through-port spectra with resonance dips.
The fit input is the measured single-bus ratio in linear power:

    T_ratio = P_device / P_single_bus

Double-bus through-port power transmission with fixed local FSR:

    phi(lambda) = 2*pi*(lambda - lambda0) / FSR

    T_ring = (t^2 + (a*t)^2 - 2*a*t^2*cos(phi))
             / (1 + (a*t^2)^2 - 2*a*t^2*cos(phi))

The local measured shape is modeled as:

    T_local(lambda) = (B0 + B1*(lambda - lambda0)) * T_ring

Here `t` is the self-coupling amplitude and `a` is the round-trip amplitude
transmission. The local background term absorbs residual measurement envelope
differences between device and single-bus traces. This model does not estimate effective index directly. FSR in
wavelength estimates group index (`ng_from_fsr`), not phase effective index.
"""

from __future__ import annotations

import numpy as np


def through_power_fixed_fsr(
    wavelength_nm: np.ndarray,
    t: float,
    a: float,
    lambda0_nm: float,
    b0: float,
    b1: float,
    fsr_nm: float,
) -> np.ndarray:
    x = np.asarray(wavelength_nm, dtype=float)
    phase = 2.0 * np.pi * (x - lambda0_nm) / fsr_nm
    c = np.cos(phase)
    numerator = t**2 + (a * t) ** 2 - 2.0 * a * t**2 * c
    denominator = 1.0 + (a * t**2) ** 2 - 2.0 * a * t**2 * c
    baseline = b0 + b1 * (x - lambda0_nm)
    return np.clip(baseline * numerator / np.clip(denominator, 1e-15, None), 1e-15, None)
