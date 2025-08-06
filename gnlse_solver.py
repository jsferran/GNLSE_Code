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


##############################################################################################
# Solver Functions
##############################################################################################
'''
args:
    Lx, Ly, Lz, Lt, <- Physical lengths, sim duration [m] [m] [m] [s]
    Nx, Ny, Nt,     <- Spatiotemporal grid counts (ints)
    deltaZ,         <- Linear step size [m]
    deltaZ_NL,      <- Nonlinear step size [m]
    lambda0,        <- Central wavelength [m]
    n2,             <- Nonlinear coefficient [m^2 / W]
    n_xyomega,      <- Frequency-dependent transverse refractive index profile
    beta0,          <- Propagation constant at lambda0 [m^-1]
    beta1,          <- First-order dispersion coefficient at lambda0 [m^-2]
    beta2,          <- Second-order dispersion coefficient at lambda0 [m^-3]
    gain_coeff,     <- Gain coefficient [m^-1]
    gain_fwhm,      <- Gain FWHM [Hz]
    t1, t2,         <- Raman response parameters [s]
    saturation_intensity, <- Saturation intensity [W/m^2]
    pml_thickness,  <- PML thickness [m]
    pml_Wmax,       <- PML maximum absorption [m^-1]    
    save_at,        <- Array of z positions to save the field at [m]
'''
def _prepare_propagation(args, A0):
    # Unpack grid
    Lx, Ly, Lz, Lt = args["Lx"], args["Ly"], args["Lz"], args["Lt"]
    Nx, Ny, Nt     = args["Nx"], args["Ny"], args["Nt"]
    dx, dy, dt     = Lx/Nx, Ly/Ny, Lt/Nt

    deltaZ    = args["deltaZ"]
    deltaZ_NL = args["deltaZ_NL"]

    steps_total = int(Lz / deltaZ)

    # Map save positions (meters) to integer step indices
    save_at_m = np.asarray(args["save_at"], dtype=float)
    save_idx  = np.rint(save_at_m / deltaZ).astype(np.int32)
    save_idx  = np.clip(save_idx, 0, max(0, steps_total - 1))
    save_idx  = np.unique(save_idx)
    save_idx  = jnp.asarray(save_idx, dtype=jnp.int32)
    save_n    = int(save_idx.size)

    # Physics constants
    c0      = 2.997_924_58e8
    lambda0 = args["lambda0"]
    f0      = c0 / lambda0
    omega0  = 2 * jnp.pi * f0
    n2      = args["n2"]
    gamma   = n2 * omega0 / c0

    beta0, beta1, beta2 = args["beta0"], args["beta1"], args["beta2"]
    gain_coeff, gain_fwhm = args["gain_coeff"], args["gain_fwhm"]
    use_gain = bool(gain_coeff != 0.0)
    t1, t2 = args["t1"], args["t2"]

    # k / ω grids
    omega = 2*jnp.pi * jnp.fft.fftfreq(Nt, dt)     # (Nt,)
    kx    = 2*jnp.pi * jnp.fft.fftfreq(Nx, dx)     # (Nx,)
    ky    = 2*jnp.pi * jnp.fft.fftfreq(Ny, dy)     # (Ny,)

    KX    = kx[:, None, None]
    KY    = ky[None, :, None]
    OMEGA = omega[None, None, :]

    # Dispersion operator D (your formulation)
    n_xyomega = args["n_xyomega"]                  # (Nx,Ny,Nt)
    n_eff_omega = n_xyomega[Nx//2, Ny//2, :]       # (Nt,)
    beta_eff = n_eff_omega[None,None,:] * (omega0 + OMEGA) / c0
    rad = beta_eff**2 - KX**2 - KY**2
    D = 1j * (jnp.sqrt(rad + 0j) - beta0 - beta1 * OMEGA - 0.5 * beta2 * OMEGA**2)
    D_kxkyomega = jnp.exp(D * deltaZ).astype(jnp.complex64)

    # Waveguide + PML factor Nprop
    pml_thickness = args["pml_thickness"]
    pml_Wmax      = args["pml_Wmax"]

    idx = jnp.arange(Nx); idy = jnp.arange(Ny)
    d_x = jnp.minimum(idx, (Nx-1)-idx)
    d_y = jnp.minimum(idy, (Ny-1)-idy)
    ramp_x = jnp.where(d_x < pml_thickness, pml_Wmax*((pml_thickness-d_x)/pml_thickness)**2, 0.0)
    ramp_y = jnp.where(d_y < pml_thickness, pml_Wmax*((pml_thickness-d_y)/pml_thickness)**2, 0.0)

    W2d = ramp_x[:,None] + ramp_y[None,:]                      # (Nx,Ny)
    PML2d  = jnp.exp(-W2d * jnp.float64(deltaZ)).astype(jnp.complex64)

    Nprop = jnp.exp(1j*beta_eff/2 * ((n_xyomega/n_eff_omega[None,None,:])**2 - 1) * deltaZ)
    

    # Raman kernel (keep in frequency domain; rfft optional later)
    hrw = _make_hrw(Nt, dt, t1, t2).astype(jnp.complex64)

    # Gain spectral envelope
    if gain_coeff == 0.0:
        gain_term = jnp.zeros_like(OMEGA, dtype=jnp.complex64)
        use_gain_flag = False
    else:
        omega_fwhm = 2.0 * jnp.pi * f0**2 / c0 * gain_fwhm
        omega_mid  = omega_fwhm / (2 * jnp.sqrt(jnp.log(2)))
        g0 = gain_coeff/2.0
        gain_term = (g0 * jnp.exp(-(OMEGA**2)/(2*omega_mid**2))).astype(jnp.complex64)
        use_gain_flag = True

    # Pack everything needed by the z-loop
    prep = dict(
        steps_total=steps_total,
        save_idx=save_idx, save_n=save_n,
        dt=dt, dx=dx, dy=dy, omega_vec=omega.astype(jnp.complex64),
        f0=f0, gamma=gamma, PML2d=PML2d,
        D_kxkyomega=D_kxkyomega.astype(jnp.complex64),
        Nprop=Nprop.astype(jnp.complex64),
        hrw=hrw, gain_term=gain_term,
        fr=args["fr"], sw=args["sw"],
        num_diffsteps_per_nlstep=int(deltaZ_NL/deltaZ),
        deltaZ_NL=deltaZ_NL,
        use_gain=use_gain_flag,
    )
    return prep


@partial(
    jax.jit,
    static_argnames=('steps_total','save_n', 'fr','sw','num_diffsteps_per_nlstep','use_gain')
)
def _propagate_scan(A0,
                    *,                             # keyword-only args below are constants for XLA
                    steps_total: int,
                    save_idx: jnp.ndarray,         # (save_n,)
                    save_n: int,
                    dt: float,
                    f0: float,
                    fr: float,
                    sw: int,
                    deltaZ_NL: float,
                    num_diffsteps_per_nlstep: int,
                    gamma: float,
                    omega_vec: jnp.ndarray,        # (Nt,)
                    D_kxkyomega: jnp.ndarray,      # (Nx,Ny,Nt)
                    Nprop: jnp.ndarray,            # (Nx,Ny,Nt)
                    PML2d,            # (Nx,Ny)
                    hrw: jnp.ndarray,              # (Nt,)
                    gain_term: jnp.ndarray,        # (Nx,Ny,Nt)
                    saturation_intensity: float,
                    use_gain: bool):
    """
    JIT-compiled z loop using lax.scan. Returns the saved field stack.
    """
    Nx, Ny, Nt = A0.shape
    save_buf0 = jnp.zeros((Nx, Ny, Nt, save_n), dtype=A0.dtype)
    save_ptr0 = jnp.array(0, dtype=jnp.int32)
    count0    = jnp.array(1, dtype=jnp.int32)

    def body(carry, i):
        field, count, save_ptr, save_buf = carry

        # one split-step (your function is already jit-compiled)
        field, count = split_step(
            field, count,
            dt=dt, f0=f0, fr=fr, sw=sw,
            deltaZ_NL=deltaZ_NL,
            num_diffsteps_per_nlstep=num_diffsteps_per_nlstep,
            gamma=gamma,
            omega_vec=omega_vec,
            D_kxkyomega=D_kxkyomega,
            Nprop=Nprop,
            PML2d = PML2d,
            hrw=hrw,
            gain_term=gain_term,
            saturation_intensity=saturation_intensity,
            use_gain=use_gain
        )

        # decide whether to save this step
        # guard against save_ptr == save_n (no more saves)
        def _do_save(args):
            field, count, save_ptr, save_buf, i = args
            save_buf = save_buf.at[..., save_ptr].set(field)
            return (field, count, save_ptr+1, save_buf)

        def _skip_save(args):
            field, count, save_ptr, save_buf, i = args
            return (field, count, save_ptr, save_buf)

        can_save   = save_ptr < save_n
        want_index = jnp.where(can_save, save_idx[save_ptr], -1)
        save_now   = jnp.logical_and(can_save, i == want_index)

        field, count, save_ptr, save_buf = jax.lax.cond(
            save_now, _do_save, _skip_save, (field, count, save_ptr, save_buf, i)
        )

        return (field, count, save_ptr, save_buf), None

    # scan over all linear steps (0..steps_total-1)
    (field_end, _, save_ptr_end, save_buf), _ = jax.lax.scan(
        body,
        (A0.astype(jnp.complex64), count0, save_ptr0, save_buf0),
        jnp.arange(steps_total, dtype=jnp.int32)
    )

    # If user asked to save more slots than visited (e.g., last step),
    # you can optionally write the final field into the remaining slot(s).
    # Here we mimic your original behavior: fill the next slot with final field.
    def _fill_tail(args):
        save_buf, field_end, save_ptr_end = args
        save_buf = save_buf.at[..., save_ptr_end].set(field_end)
        return save_buf
    save_buf = jax.lax.cond(save_ptr_end < save_n, _fill_tail, lambda x: x[0],
                            (save_buf, field_end, save_ptr_end))

    return save_buf  # shape (Nx,Ny,Nt,save_n)

def GNLSE3D_propagate_scan(args, A0):
    prep = _prepare_propagation(args, A0)

    # Run once to compile; then time a second run (common pattern)
    _ = _propagate_scan(
        A0,
        steps_total=prep["steps_total"],
        save_idx=prep["save_idx"], save_n=prep["save_n"],
        dt=prep["dt"], f0=prep["f0"],
        fr=prep["fr"], sw=prep["sw"],
        deltaZ_NL=prep["deltaZ_NL"],
        num_diffsteps_per_nlstep=prep["num_diffsteps_per_nlstep"],
        gamma=prep["gamma"],
        omega_vec=prep["omega_vec"],
        D_kxkyomega=prep["D_kxkyomega"],
        Nprop=prep["Nprop"],
        PML2d=prep["PML2d"],
        hrw=prep["hrw"],
        gain_term=prep["gain_term"],
        saturation_intensity=args["saturation_intensity"],
        use_gain=prep["use_gain"]
    ).block_until_ready()

    t0 = time.time()
    field_saved = _propagate_scan(
        A0,
        steps_total=prep["steps_total"],
        save_idx=prep["save_idx"], save_n=prep["save_n"],
        dt=prep["dt"], f0=prep["f0"],
        fr=prep["fr"], sw=prep["sw"],
        deltaZ_NL=prep["deltaZ_NL"],
        num_diffsteps_per_nlstep=prep["num_diffsteps_per_nlstep"],
        gamma=prep["gamma"],
        omega_vec=prep["omega_vec"],
        D_kxkyomega=prep["D_kxkyomega"],
        Nprop=prep["Nprop"],
        PML2d=prep["PML2d"],
        hrw=prep["hrw"],
        gain_term=prep["gain_term"],
        saturation_intensity=args["saturation_intensity"],
        use_gain=prep["use_gain"]
    ).block_until_ready()
    elapsed = time.time() - t0

    return dict(field=field_saved, dt=prep["dt"], dx=prep["dx"], seconds=elapsed)

    
def _dA_dz_NL( A_xy_t,
           *, dt,
           f0, # Hz
           fr, # Raman fraction
           sw, # self-steepening flag (0/1)
           gamma, # W^-1 m
           omega_vec, # (Nt, Nx, Ny)
           hrw,       # (Nt,)
           gain_term,  # (Nt, Nx, Ny)
           saturation_intensity, # J/m^2
           use_gain: bool
         ):
    
    Nx, Ny, Nt = A_xy_t.shape;

    # Kerr + Raman:
    absA2_xy_t = jnp.abs(A_xy_t)**2;

    if fr == 0.0:
        NL_core_xy_t = absA2_xy_t * A_xy_t
    else:
        # FFT along time
        I_xy_omega = jnp.fft.fft(absA2_xy_t, axis=2)                # (Nx,Ny,Nt)
        H_omega    = hrw[None, None, :]                              # broadcast
        Raman_xy_t = jnp.fft.ifft(H_omega * I_xy_omega, axis=2)      # (Nx,Ny,Nt)
        NL_core_xy_t = ((1.0 - fr) * absA2_xy_t + fr * Raman_xy_t) * A_xy_t;


    # Self-steepening:
    if sw == 1:
        NL_core_xy_omega = jnp.fft.fft(NL_core_xy_t, axis=2);

        NL_core_xy_omega *= (1.0 + omega_vec[None, None, :] / (2.0 * jnp.pi * f0));
        NL_core_xy_t = jnp.fft.ifft(NL_core_xy_omega, axis=2);
    
    dA_xy_t = 1j * gamma * NL_core_xy_t;


    # Saturable gain:
    def _add_gain(args):
        dA_xy_t, A_xy_t, absA2_xy_t = args
        power_xy = jnp.sum(absA2_xy_t, axis=2) * dt
        gain_pref = 1.0 / (1.0 + power_xy / saturation_intensity)
        gain_pref = gain_pref[:, :, None]
        A_xy_omega = jnp.fft.fft(A_xy_t, axis=2)
        A_gain_xy_omega = gain_term * (gain_pref * A_xy_omega)
        return dA_xy_t + jnp.fft.ifft(A_gain_xy_omega, axis=2)
   
    dA_xy_t = jax.lax.cond(
        use_gain,
        _add_gain,
        lambda args: dA_xy_t,
        (dA_xy_t, A_xy_t, absA2_xy_t)
    );

    return dA_xy_t;
    



@partial(jax.jit, static_argnames=('fr','sw','num_diffsteps_per_nlstep', 'use_gain'))
def split_step( field_xy_t,
                count: int,
                *,
                dt : float,
                f0 : float,
                fr : float,
                sw : int,
                deltaZ_NL : float,
                num_diffsteps_per_nlstep: int,
                gamma: float,                     # W^-1 m
                omega_vec,                       # (Nt,Nx,Ny)
                D_kxkyomega,                      # (Nx,Ny,Nt)
                Nprop,                            # (Nx,Ny,Nt)
                PML2d,
                hrw,                              # (Nt,)
                gain_term,                        # (Nt,Nx,Ny)
                saturation_intensity: float,
                use_gain : bool
                ):
    Nx, Ny, Nt = jnp.shape(field_xy_t);

    
    # Linear operation of D:
    field_k = jnp.fft.fftn(field_xy_t, axes=(0,1,2));  # (Nx,Ny,Nt)
    field_k *= D_kxkyomega;
    field_xy_t = jnp.fft.ifftn(field_k, axes=(0,1,2));

    # Waveguide + PML factors:
    field_xy_t *= Nprop;
    field_xy_t *= PML2d[:, :, None];  # (Nx,Ny,Nt)


    ## Nonlinear steps:
    def _do_nl(operands):
        field, count = operands
        dA = deltaZ_NL * _dA_dz_NL(
            field, dt=dt, f0=f0, fr=fr, gamma=gamma, sw=sw,
            omega_vec=omega_vec, hrw=hrw, gain_term=gain_term,
            saturation_intensity=saturation_intensity, use_gain=use_gain
        )
        # reset counter to 1 with the SAME dtype as `count`
        return field + dA, jnp.asarray(1, dtype=count.dtype)

    def _skip_nl(operands):
        field, count = operands
        # increment by 1 with the SAME dtype
        return field, count + jnp.asarray(1, dtype=count.dtype)

    # ensure the rhs of the comparison has the SAME dtype as `count`
    trigger = count == jnp.asarray(num_diffsteps_per_nlstep, dtype=count.dtype)
    field_xy_t, new_count = jax.lax.cond(
        trigger, _do_nl, _skip_nl, (field_xy_t, count)
    )

    return field_xy_t, new_count


def _make_hrw(Nt: int, dt: float, t1 = 12.2e-15, t2 = 32.0e-15, *, dtype=jnp.float64):
    """Raman response H(ω)."""
    t  = dt * jnp.arange(Nt, dtype=dtype);
    hr = ((t1**2 + t2**2)/(t1 * t2**2)) * jnp.exp(-t/t2) * jnp.sin(t/t1);
    return jnp.fft.ifft(hr) * Nt;
