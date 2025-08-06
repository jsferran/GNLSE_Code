import numpy as np
import jax
import jax.numpy as jnp

def compute_self_focusing_pixel_intensity(P_launch,
                                          n2,
                                          dx,
                                          dy,
                                          *,
                                          safety=0.5,
                                          include_delta_n_cap=True,
                                          n0=1.45,
                                          delta_n_frac=1e-2,
                                          lambda0=None):
    """
    Conservative per-pixel intensity threshold [W/m^2] beyond which a (x,y,t,z) GNLSE
    simulation is likely physically/numerically unreliable (collapse under-resolved).

    Parameters
    ----------
    P_launch : float
        Launched power [W].
    n2 : float
        Kerr coefficient [m^2/W].
    dx, dy : float
        Transverse pixel sizes [m].
    safety : float, optional
        Fraction of the single-pixel power concentration you are willing to tolerate
        before declaring 'under-resolved'. Must be in (0, 1]. Default 0.5.
    include_delta_n_cap : bool, optional
        If True, also cap intensity by requiring Δn = n2 * I <= delta_n_frac * n0.
        Default True.
    n0 : float, optional
        Linear refractive index used in the Δn cap. Default 1.45.
    delta_n_frac : float, optional
        Fractional bound for Δn/n0 in the Δn cap (e.g., 1e-2 = 1%). Default 1e-2.
    lambda0 : float or None, optional
        If provided [m], the result will also include the (bulk) critical power estimate
        P_cr ≈ 0.148 * lambda0^2 / (n0 * n2) for context (not used to set the threshold).

    Returns
    -------
    out : dict
        {
          'I_thresh': float,                 # threshold [W/m^2]
          'components': {
              'pixel_cap': float,            # safety * P_launch / (dx*dy)
              'delta_n_cap': float or None,  # (delta_n_frac * n0) / n2  (if used)
          },
          'context': {
              'area_pixel': float,           # dx*dy [m^2]
              'P_cr': float or None          # if lambda0 provided, [W]
          }
        }
    """
    # Basic checks
    if not (dx > 0 and dy > 0):
        raise ValueError("dx and dy must be positive.")
    if not (0 < safety <= 1.0):
        raise ValueError("safety must be in (0, 1].")
    if include_delta_n_cap and not (delta_n_frac > 0):
        raise ValueError("delta_n_frac must be positive when include_delta_n_cap=True.")

    area_pix = dx * dy
    pixel_cap = safety * (P_launch / area_pix)            # W/m^2

    if include_delta_n_cap:
        delta_n_cap = (delta_n_frac * n0) / n2            # W/m^2
        I_thresh = float(min(pixel_cap, delta_n_cap))
    else:
        delta_n_cap = None
        I_thresh = float(pixel_cap)

    # Optional critical power (context only; not used in the threshold)
    if lambda0 is not None:
        P_cr = 0.148 * (lambda0**2) / (n0 * n2)           # W   (Gaussian beam, bulk)
    else:
        P_cr = None

    return {
        'I_thresh': I_thresh,
        'components': {
            'pixel_cap': float(pixel_cap),
            'delta_n_cap': (float(delta_n_cap) if delta_n_cap is not None else None),
        },
        'context': {
            'area_pixel': float(area_pix),
            'P_cr': (float(P_cr) if P_cr is not None else None),
        }
    }


def max_intensity_event(field_xyt, z, payload):
    Imax = jnp.max(jnp.abs(field_xyt)**2)
    return Imax > jnp.asarray(payload['I_thresh'], dtype=field_xyt.real.dtype);