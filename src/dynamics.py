"""
dynamics.py
Canonical discrete-time linearisation of the Dufour autoclave plant.

This is the SINGLE source of truth for the linearised state-space (Ap, Bp) used
to build the per-step MPC-QP. Both the CVXPY/OSQP baseline and the SNN-QP
controller call `linearize(...)`, so the bulk of the model is provably identical
across the two pipelines (previously each hand-coded its own copy, which could
silently drift).

Two regularisation tiers, both documented for reproducibility / deviation
tracking:

  * Tier-2 (always on, mathematically necessary): the cure Jacobian
    J_alpha = f0 * (m/alpha - n/(1-alpha)) has a genuine pole as alpha -> 0
    (since m < 1, f0 * m/alpha ~ alpha^(m-1) -> inf). The derivative simply
    cannot be evaluated at alpha = 0 or 1, so we evaluate it at a point clamped
    to [ALPHA_FLOOR, ALPHA_CEIL]. This regularises the *evaluation point* of the
    derivative, not the resulting finite value. Applied identically in BOTH
    controllers.

  * Tier-3 (opt-in via `trust_region=True`): a bound on the magnitude of the
    exotherm Jacobian contributions injected into Ap. This is NOT physical -- it
    is a trust region on the linearisation. It exists because the SNN-QP uses a
    *condensed* dense prediction (Ap^k over the horizon), and the plant's
    degree-of-cure states are eigenvalue-1 integrators that are UNCONTROLLABLE
    from the autoclave temperature (you cannot un-cure resin). Prestabilisation
    -- the standard remedy for unstable prediction models -- is therefore
    inapplicable, and the un-regularised condensed QP is too ill-conditioned for
    the projection-based SNN solver to keep feasible (verified empirically: it
    diverges to ~1e12). The OSQP baseline solves the sparse form and needs no
    such bound, so it is called with `trust_region=False`. This single, named
    flag is the ONLY modelling difference between the two controllers, and it is
    the honest, documented deviation from a pure first-order linearisation.
"""
import numpy as np
import src.constants as const

ALPHA_FLOOR = 1e-3
ALPHA_CEIL = 0.999
NX = 10  # 3 composite temps + 4 tooling temps + 3 cure states


def linearize(T0_degC, alpha0, trust_region=False):
    """Return (Ap, Bp): the discrete-time linearised plant about (T0, alpha0).

    Parameters
    ----------
    trust_region : bool
        False (baseline): pure first-order model, Tier-2 clamp only.
        True (SNN-QP): additionally bound the exotherm Jacobian terms so the
        condensed dense QP stays solvable by the projection solver. See module
        docstring for the justification.
    """
    Ap = np.zeros((NX, NX))
    Bp = np.zeros(NX)
    T0_K = T0_degC + 273.15

    f0 = const.AC * np.exp(-const.EA / (const.R * T0_K)) * \
        (alpha0 ** const.M_EXP) * ((1 - alpha0) ** const.N_EXP)

    dT = f0 * (const.EA / (const.R * (T0_K ** 2)))
    a_safe = max(ALPHA_FLOOR, min(alpha0, ALPHA_CEIL))     # Tier-2 (necessary)
    da = f0 * ((const.M_EXP / a_safe) - (const.N_EXP / (1 - a_safe)))

    exo_mult = (const.MR * const.DH / const.CPC) * const.TE

    # Exotherm Jacobian contributions -- pure model, or Tier-3 trust-region bound
    if trust_region:
        exo_dT = min(exo_mult * dT, const.FC * 1.8)
        exo_da = float(np.clip(exo_mult * da, -const.FC * 3.0, const.FC * 3.0))
        da_self = (1.0 - 1e-4) + float(np.clip(da * const.TE, -0.90, 0.0))
        dT_cross = float(np.clip(dT * const.TE, 0.0, 0.1))
    else:
        exo_dT = exo_mult * dT
        exo_da = exo_mult * da
        da_self = 1.0 + da * const.TE
        dT_cross = dT * const.TE

    # Composite nodes: heat diffusion + exothermic self-heating
    for i in range(const.NZ_C):
        Ap[i, i] = 1 - 2 * const.FC + exo_dT
        Ap[i, i + 7] = exo_da
        if i == 0:
            Ap[i, i + 1] = 2 * const.FC                    # symmetry axis at the centre
        elif i == const.NZ_C - 1:
            Ap[i, i - 1] = const.FC
            Ap[i, const.NZ_C] = const.FC                   # composite/tooling interface
        else:
            Ap[i, i - 1] = const.FC
            Ap[i, i + 1] = const.FC

    # Tooling nodes: pure heat diffusion; Ta enters at the outer node
    for i in range(const.NZ_C, const.NZ_C + const.NZ_T):
        Ap[i, i] = 1 - 2 * const.FT
        if i == const.NZ_C:
            Ap[i, i - 1] = const.FT
            Ap[i, i + 1] = const.FT
        elif i == const.NZ_C + const.NZ_T - 1:
            Ap[i, i - 1] = const.FT
            Bp[i] = const.FT                                # INPUT: autoclave air Ta
        else:
            Ap[i, i - 1] = const.FT
            Ap[i, i + 1] = const.FT

    # Cure dynamics: alpha depends on alpha and on temperature
    for i in range(7, 10):
        Ap[i, i] = da_self
        Ap[i, i - 7] = dT_cross

    return Ap, Bp
