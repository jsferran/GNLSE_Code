import time;
from math import pi, sqrt;

import jax;
import jax.numpy as jnp;
from types import SimpleNamespace;
from functools import partial;
from pathlib import Path;
import numpy as np;

from scipy.sparse.linalg import eigsh, LinearOperator, cg;

jax.config.update("jax_enable_x64", True);



######################################################################################################
# Source Builder Functions
######################################################################################################
    
    
########################
### Construct HG Modes
########################

def precompute_H_table(n_max: int, 
                       x: np.ndarray, 
                       w0: float) -> np.ndarray:
    xi = np.sqrt(2.0) * x / w0;
    H = np.empty((n_max + 1, x.size), dtype=x.dtype);

    H[0] = 1.0;
    if n_max >= 1:
        H[1] = 2.0 * xi;
    for n in range(2, n_max + 1):
        H[n] = 2.0 * xi * H[n-1] - 2.0 * (n-1) * H[n-2];
    return H;

def build_hg_library(N: int,
                    x: np.ndarray,
                    y: np.ndarray,
                    folder: str | Path,
                    w0: float = 50e-6,
                    ) -> None:
    #
    out_folder = Path(folder);
    out_folder.mkdir(parents = True, exist_ok = True);

    p_max = int(np.floor(np.sqrt(N-1)));
    q_max = 2 * p_max;
    n_max = q_max;
    Hx = precompute_H_table(n_max, x, w0);
    Hy = precompute_H_table(n_max, y, w0);

    gx = np.exp(-(x**2) / w0**2);
    gy = np.exp(-(y**2) / w0**2);

    dx = x[1] - x[0];
    dy = y[1] - y[0]
    area = dx * dy;

    normx =  np.sqrt(np.sum((Hx*gx)**2, axis = 1)*dx);
    normy =  np.sqrt(np.sum((Hy*gy)**2, axis = 1)*dy);

    for idx in range(N):
        p = int(np.floor(np.sqrt(idx)));
        q = idx - p * p;

        env = (Hx[p][:, None] * Hy[q][None,:]) * (gx[:, None] * gy[None, :]);
        env /= normx[p] * normy[q];
        np.savez(out_folder / f"HG_{idx:04d}.npz", field=env.astype(np.complex128));
        

#################################################################
### Solve Modes and Prop-constants From Arbitrary Index Profile
#################################################################



'''
Takes a transverse refractive index profile and (central) wavelength to determine modes:
'''

def solve_modes(n_xy,
                n_ref: float = 1.0,
                x = [0],
                y = [0],
                lambda0: float = 1030e-9,
                n_modes: int = 6, 
                folder: str | Path = "modes",
                maxiter: int = 1500,
                tol: float = 1e-9):

    #Make/check for folder:
    folder = Path(folder);
    folder.mkdir(exist_ok=True, parents=True);

    nx, ny = len(x), len(y);
    dx, dy = float(x[1]-x[0]), float(y[1]-y[0]);
    k0 = 2.0 * np.pi / lambda0;
    beta0 = k0 * n_ref;

    
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx);
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy);

    KX, KY = np.meshgrid(kx, ky, indexing="ij");
    lap_symbol = -(KX**2 + KY**2);

    n2_xy = n_xy**2                              # square once
    k2_n2 = (k0**2 * n2_xy).astype(np.float64)   # (Nx,Ny)


    #Build the Helmholtz operator:
    def matvec(vec: np.ndarray) -> np.ndarray:
        psi = vec.reshape((nx, ny));
        lap_psi = np.fft.ifft2(lap_symbol * np.fft.fft2(psi));
        return (lap_psi + k2_n2 * psi).ravel();

    H = LinearOperator(
        shape=(nx*ny, nx*ny),
        matvec=matvec,
        dtype=np.complex128,
    );

    
    #Solve for eigen-modes and values, sorted from high to low:
    beta2s, eigvecs = eigsh(
        H, k=n_modes, which="LA",
        tol=tol, maxiter=maxiter, ncv=max(2*n_modes+1, 40),
    );
    
    order = np.argsort(beta2s)[::-1]; #
    beta2s, eigvecs = beta2s[order], eigvecs[:, order];

    beta2s_shifted = beta2s + k0**2 * n_ref**2 ;
    

    #Normalize and save the modes:
    modes = [];
    area = dx * dy;
    for m, (beta2, vec) in enumerate(zip(beta2s_shifted, eigvecs.T)):
        
        beta = float(np.sqrt(beta2.real));
        field = vec.reshape((nx, ny)).astype(np.complex128);
        field /= np.sqrt(np.sum(np.abs(field)**2)* area);
        modes.append((beta, field));

        np.savez(
            folder / f"mode_{m:04d}.npz",
            beta=beta,
            field=field,
            x=np.asarray(x),
            y=np.asarray(y),
        );

    return modes;

#####################################################################
### Create Fields from Mode Files
#####################################################################
def make_source_from_files(folder, heading="mode", file_format="npz", weights=None):
    if weights is None: weights = {};
    indices = list(weights.keys());
    field = None;
    coeffs = [];
    for i in indices:
        mode = np.load(f"{folder}/{heading}_{i:04d}.{file_format}", mmap_mode="r")["field"];
        ci   = weights[i];
        coeffs.append(ci);
        field = ci * mode if field is None else field + ci * mode;
    return jnp.asarray(field), jnp.asarray(indices), jnp.asarray(coeffs);


def norm_scale_field_weights(field, indices, coeffs, power, dx, dy):
    # compute real power, not coeff norm
    P_now = jnp.sum(jnp.abs(field)**2) * dx * dy;
    s = jnp.sqrt(power / P_now);
    field  = field * s;
    coeffs = coeffs * s;
    return field, dict(zip(indices.tolist(), coeffs.tolist()));
    
####################################################################
### Temporal Profiles
####################################################################
def cw_temp_profile(lambdas, phis, Lt, Nt):
    lambdas = jnp.atleast_1d(lambdas)          # make them 1‑D
    phis    = jnp.atleast_1d(phis)

    c0     = 2.99792458e8
    omegas = 2 * jnp.pi * c0 / lambdas         # (M,)
    dt     = Lt / Nt
    t      = jnp.linspace(-Lt/2, Lt/2 - dt, Nt)

    E_t = jnp.sum(jnp.exp(1j * (omegas[:, None] * t + phis[:, None])), axis=0)
    return E_t.astype(jnp.complex64) 
    
def gaussian_pulse_profile(t0, fwhm, Lt, Nt, carrier_omega=0.0, phase=0.0):
    dt   = Lt / Nt;
    t    = jnp.linspace(-Lt/2, Lt/2 - dt, Nt);
    sigma= fwhm / (2 * jnp.sqrt(2 * jnp.log(2)));   # sigma from FWHM
    envelope = jnp.exp(-0.5 * ((t - t0) / sigma) ** 2);
    carrier  = jnp.exp(1j * (carrier_omega * t + phase));
    return (envelope * carrier).astype(jnp.complex128);

def combine_spatial_temporal(E_xy, E_t):
    return E_xy[:, :, None] * E_t[None, None, :];


# ---------- utilities ----------

def _fft_omega_grid(Lt: float, Nt: int):
    """Angular frequency grid matching jnp.fft.fft/ifft conventions."""
    dt = Lt / Nt
    omega = 2 * jnp.pi * jnp.fft.fftfreq(Nt, d=dt)  # [0.. +, -, .. -]
    return omega, dt

def _freq_to_bin(omega: float, Lt: float, Nt: int):
    """Nearest FFT bin index for a given angular frequency."""
    # k ≈ round(ω * Lt / (2π))
    k = int(jnp.rint(omega * Lt / (2 * jnp.pi)))
    # wrap to valid Python index range [-Nt/2..Nt/2-1] in FFT ordering
    k_mod = (k % Nt)
    return k_mod

# ---------- profiles built in frequency ----------

def cw_temp_profile_freq(lambdas, phis, Lt, Nt, amplitudes=None, dtype=jnp.complex128):
    """
    Multi-tone CW built in frequency domain with exact bin placement.
    Each tone has amplitude=1 (or given), phase phi, at ω=2π c / λ.
    Returns E_t on Nt samples spanning Lt.
    """
    c0 = 2.99792458e8
    lambdas = jnp.atleast_1d(lambdas)
    phis    = jnp.atleast_1d(phis)
    assert lambdas.shape == phis.shape, "lambdas and phis must have same length."

    M = lambdas.size
    if amplitudes is None:
        amplitudes = jnp.ones(M, dtype=dtype)
    else:
        amplitudes = jnp.atleast_1d(amplitudes).astype(dtype)
        assert amplitudes.shape == (M,)

    omega, _ = _fft_omega_grid(Lt, Nt)
    Ew = jnp.zeros(Nt, dtype=dtype)

    # Place each line on the nearest FFT bin exactly
    for m in range(M):
        omega_m = 2 * jnp.pi * c0 / lambdas[m]
        k = _freq_to_bin(omega_m, Lt, Nt)
        # Set the complex bin amplitude. Because ifft divides by Nt,
        # we multiply by Nt so a single bin gives a tone with 'amplitude' in time.
        Ew = Ew.at[k].set(Ew[k] + amplitudes[m] * jnp.exp(1j * phis[m]) * Nt)

    Et = jnp.fft.ifft(Ew)    # time-domain complex envelope
    return Et.astype(dtype)

def gaussian_pulse_profile_freq(t0, fwhm, Lt, Nt, *,
                                carrier_omega=0.0, phase=0.0,
                                center_in_window=True,
                                dtype=jnp.complex128):
    """
    Transform-limited Gaussian built in frequency domain on a [0, Lt) grid with Nt samples.
    If center_in_window=True, the pulse peak is at t = Lt/2 (so it doesn't wrap).
    """
    omega, dt = _fft_omega_grid(Lt, Nt)  # omega = 2π * fftfreq(Nt, dt)
    sigma_t = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))

    # place the Gaussian in frequency at carrier_omega
    # time shift by t_shift: multiply by exp(-i ω t_shift)
    t_shift = t0 + (0.5 * Lt if center_in_window else 0.0)

    Ew = jnp.exp(-0.5 * (sigma_t**2) * (omega - carrier_omega)**2) \
         * jnp.exp(-1j * omega * t_shift) * jnp.exp(1j * phase)

    Et = jnp.fft.ifft(Ew)
    Et = Et / jnp.max(jnp.abs(Et))  # unit peak
    return Et.astype(dtype)

