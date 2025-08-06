import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)  # ensures float64/complex128 kernels

##############################################################################################
# Solver Functions — complex128 everywhere
##############################################################################################

def _make_hrw(Nt: int, dt: float, t1 = 12.2e-15, t2 = 32.0e-15, *, dtype=jnp.float64):
    """Raman response H(ω) as complex128."""
    t  = dt * jnp.arange(Nt, dtype=dtype)
    hr = ((t1**2 + t2**2)/(t1 * t2**2)) * jnp.exp(-t/t2) * jnp.sin(t/t1)
    return jnp.fft.ifft(hr).astype(jnp.complex128) * Nt  # complex128


def _dA_dz_NL_rest(A_xy_t,
                   *, dt: float, f0: float, fr: float, sw: int, gamma: float,
                   omega_vec: jnp.ndarray, hrw: jnp.ndarray, gain_term: jnp.ndarray,
                   saturation_intensity: float, use_gain: bool) -> jnp.ndarray:
    """
    Residual NL derivative (complex128): Raman + self-steepening + gain/saturation.
    NOTE: The instantaneous Kerr i*γ|A|^2 A is handled exactly elsewhere and is NOT included here.
    """
    Nx, Ny, Nt = A_xy_t.shape
    absA2_xy_t = jnp.abs(A_xy_t)**2
    NL_core_xy_t = jnp.zeros_like(A_xy_t, dtype=jnp.complex128)

    # Raman (convolution in time)
    if fr != 0.0:
        I_xy_omega = jnp.fft.fft(absA2_xy_t, axis=2).astype(jnp.complex128)   # (Nx,Ny,Nt)
        H_omega    = hrw[None, None, :]                                        # (1,1,Nt)
        Raman_xy_t = jnp.fft.ifft(H_omega * I_xy_omega, axis=2)                # (Nx,Ny,Nt)
        NL_core_xy_t = NL_core_xy_t + (fr * Raman_xy_t) * A_xy_t

    # Self-steepening (acts on NL_core)
    if sw == 1:
        NL_core_xy_omega = jnp.fft.fft(NL_core_xy_t, axis=2)
        NL_core_xy_omega *= (1.0 + omega_vec[None, None, :] / (2.0 * jnp.pi * f0))
        NL_core_xy_t = jnp.fft.ifft(NL_core_xy_omega, axis=2)

    dA_xy_t = (1j * gamma) * NL_core_xy_t  # complex128

    # Saturable gain with spectral envelope gain_term(ω)
    def _add_gain(args):
        dA_xy_t, A_xy_t, absA2_xy_t = args
        power_xy  = jnp.sum(absA2_xy_t, axis=2) * dt                     # (Nx,Ny) float64
        gain_pref = 1.0 / (1.0 + power_xy / saturation_intensity)        # (Nx,Ny)
        gain_pref = gain_pref[:, :, None]                                 # (Nx,Ny,1)
        A_xy_omega = jnp.fft.fft(A_xy_t, axis=2)
        # gain_term is (1,1,Nt) or (Nx,Ny,Nt), both broadcast → complex128
        A_gain_xy_omega = gain_term * (gain_pref * A_xy_omega)
        return dA_xy_t + jnp.fft.ifft(A_gain_xy_omega, axis=2)

    dA_xy_t = jax.lax.cond(
        use_gain,
        _add_gain,
        lambda args: dA_xy_t,
        (dA_xy_t, A_xy_t, absA2_xy_t)
    )
    return dA_xy_t.astype(jnp.complex128)


def _prepare_propagation(args, A0):
    """Precompute grids and half-step propagators. All complex arrays are complex128."""
    # Grid
    Lx, Ly, Lz, Lt = args["Lx"], args["Ly"], args["Lz"], args["Lt"]
    Nx, Ny, Nt     = args["Nx"], args["Ny"], args["Nt"]
    dx, dy, dt     = Lx/Nx, Ly/Ny, Lt/Nt

    deltaZ    = float(args["deltaZ"])
    deltaZ_NL = float(args["deltaZ_NL"])
    steps_total = int(round(Lz / deltaZ))

    # Save indices
    save_at_m = np.asarray(args["save_at"], dtype=float)
    save_idx  = np.rint(save_at_m / deltaZ).astype(np.int32)
    save_idx  = np.clip(save_idx, 0, max(0, steps_total - 1))
    save_idx  = np.unique(save_idx)
    save_idx  = jnp.asarray(save_idx, dtype=jnp.int32)
    save_n    = int(save_idx.size)

    # Physics constants
    c0      = 2.997_924_58e8
    lambda0 = float(args["lambda0"])
    f0      = c0 / lambda0
    omega0  = 2 * jnp.pi * f0
    n2      = float(args["n2"])
    gamma   = n2 * omega0 / c0

    beta0, beta1, beta2 = float(args["beta0"]), float(args["beta1"]), float(args["beta2"])
    gain_coeff, gain_fwhm = float(args["gain_coeff"]), float(args["gain_fwhm"])
    use_gain = bool(gain_coeff != 0.0)
    t1, t2 = float(args["t1"]), float(args["t2"])

    # k / ω grids (float64)
    omega = 2*jnp.pi * jnp.fft.fftfreq(Nt, dt)     # (Nt,) float64
    kx    = 2*jnp.pi * jnp.fft.fftfreq(Nx, dx)     # (Nx,)
    ky    = 2*jnp.pi * jnp.fft.fftfreq(Ny, dy)     # (Ny,)

    KX    = kx[:, None, None]                      # (Nx,1,1)
    KY    = ky[None, :, None]                      # (1,Ny,1)
    OMEGA = omega[None, None, :]                   # (1,1,Nt)

    # Material index (x,y,ω)
    n_xyomega = args["n_xyomega"]                  # expected (Nx,Ny,Nt) float64 or complex
    n_xyomega = jnp.asarray(n_xyomega, dtype=jnp.float64)
    n_eff_omega = n_xyomega[Nx//2, Ny//2, :]       # (Nt,)
    beta_eff = n_eff_omega[None,None,:] * (omega0 + OMEGA) / c0  # (1,1,Nt)

    # Spectral generator D(kx,ky,ω) and half-step propagator
    rad = beta_eff**2 - KX**2 - KY**2
    D = 1j * (jnp.sqrt(rad + 0.0) - beta0 - beta1*OMEGA - 0.5*beta2*OMEGA**2)  # complex128
    D_half = jnp.exp(D * (deltaZ/2)).astype(jnp.complex128)                     # (Nx,Ny,Nt)

    # PML (x,y) and waveguide phase (x,y,ω)
    pml_thickness = int(args["pml_thickness"])
    pml_Wmax      = float(args["pml_Wmax"])

    idx = jnp.arange(Nx); idy = jnp.arange(Ny)
    d_x = jnp.minimum(idx, (Nx-1)-idx)
    d_y = jnp.minimum(idy, (Ny-1)-idy)
    ramp_x = jnp.where(d_x < pml_thickness, pml_Wmax*((pml_thickness-d_x)/pml_thickness)**2, 0.0)
    ramp_y = jnp.where(d_y < pml_thickness, pml_Wmax*((pml_thickness-d_y)/pml_thickness)**2, 0.0)
    W2d = ramp_x[:,None] + ramp_y[None,:]                                     # (Nx,Ny)
    PML_half = jnp.exp(-W2d * jnp.float64(deltaZ/2)).astype(jnp.complex128)   # (Nx,Ny)

    Nprop_half = jnp.exp(
        1j*beta_eff/2 * ((n_xyomega/n_eff_omega[None,None,:])**2 - 1.0) * (deltaZ/2)
    ).astype(jnp.complex128)                                                  # (Nx,Ny,Nt)

    # Raman kernel (complex128)
    hrw = _make_hrw(Nt, dt, t1, t2).astype(jnp.complex128)

    # Gain spectral envelope g(ω) (use Hz→rad/s: ω_FWHM = 2π * f_FWHM)
    if use_gain:
        omega_fwhm = 2.0 * jnp.pi * gain_fwhm
        omega_mid  = omega_fwhm / (2.0 * jnp.sqrt(jnp.log(2.0)))
        g0 = gain_coeff/2.0
        gain_term = (g0 * jnp.exp(-(OMEGA**2)/(2*omega_mid**2))).astype(jnp.complex128)  # (1,1,Nt)
        use_gain_flag = True
    else:
        gain_term = jnp.zeros((1,1,Nt), dtype=jnp.complex128)
        use_gain_flag = False

    # Residual NL substeps (inside NL stage)
    m_nl = int(args.get("m_nl_substeps", 1))

    prep = dict(
        steps_total=steps_total,
        save_idx=save_idx, save_n=save_n,
        dt=jnp.float64(dt), dx=jnp.float64(dx), dy=jnp.float64(dy),
        omega_vec=omega.astype(jnp.float64),
        f0=jnp.float64(f0), gamma=jnp.float64(gamma),
        D_half=D_half, PML_half=PML_half, Nprop_half=Nprop_half,
        hrw=hrw, gain_term=gain_term,
        fr=float(args["fr"]), sw=int(args["sw"]),
        deltaZ_NL=jnp.float64(deltaZ_NL),
        use_gain=use_gain_flag,
        m_nl_substeps=m_nl,
    )
    return prep

# --- Adaptive sharding utilities ---------------------------------------------
import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def _best_1d_factor(n_devices: int, n_axis: int) -> int:
    """Largest divisor of n_devices that also divides n_axis (>=1)."""
    best = 1
    for d in range(1, n_devices + 1):
        if (n_devices % d == 0) and (n_axis % d == 0):
            best = d
    return best

def _make_mesh_for_time_axis(Nt: int):
    """Create a 1D mesh for sharding along the t/omega axis; fall back to replication."""
    devs = jax.devices()
    ndev = len(devs)
    if ndev == 0:
        raise RuntimeError("No JAX devices available.")
    n_shards_t = _best_1d_factor(ndev, Nt)
    # Use exactly n_shards_t devices (leave extras idle if any).
    mesh_arr = np.array(devs[:max(1, n_shards_t)]).reshape((max(1, n_shards_t),))
    mesh = Mesh(mesh_arr, axis_names=('t',))
    shard_t   = NamedSharding(mesh, P(None, None, 't'))   # shard along Nt (x,y,t)
    replicate = NamedSharding(mesh, P(None, None, None))  # fully replicated
    return mesh, shard_t, replicate

import jax
import jax.numpy as jnp

def make_propagate_while_sharded(shard_t, replicate,
                                 *,
                                 event_fn=None,
                                 stop_on_event=True,
                                 event_check_every: int = 1):
    """
    Returns a jitted function that runs a z-loop with optional early stopping on `event_fn`.
    If `event_fn` is None or stop_on_event=False, no event code is traced at all.
    """
    use_event   = bool(callable(event_fn) and stop_on_event)
    check_every = int(max(1, event_check_every))

    def _materialize_xyt_from_kwo(field_kwo):
        field_kwo = jax.lax.with_sharding_constraint(field_kwo, replicate)
        field_xyw = jnp.fft.ifftn(field_kwo, axes=(0,1)).astype(jnp.complex128)
        field_xyt = jnp.fft.ifft(field_xyw, axis=2).astype(jnp.complex128)
        return field_xyt

    @partial(
        jax.jit,
        static_argnames=('steps_total','save_n','fr','sw','use_gain','m_nl_substeps')
    )
    def _propagate_while_sharded(A0_kwo: jnp.ndarray,
                                 *,
                                 payload,                 # PyTree
                                 steps_total: int,
                                 save_idx: jnp.ndarray, save_n: int,
                                 dt: float, f0: float,
                                 fr: float, sw: int,
                                 deltaZ_NL: float,
                                 gamma: float,
                                 omega_vec: jnp.ndarray,
                                 D_half: jnp.ndarray,
                                 PML_half: jnp.ndarray,
                                 Nprop_half: jnp.ndarray,
                                 hrw: jnp.ndarray,
                                 gain_term: jnp.ndarray,
                                 saturation_intensity: float,
                                 use_gain: bool,
                                 m_nl_substeps: int):

        field_kwo0 = jax.lax.with_sharding_constraint(A0_kwo, shard_t)
        Nx, Ny, Nt = A0_kwo.shape

        save_buf0 = jnp.zeros((Nx, Ny, Nt, save_n), dtype=jnp.complex128)
        save_ptr0 = jnp.array(0, dtype=jnp.int32)
        i0        = jnp.array(0, dtype=jnp.int32)
        done0     = jnp.array(False)
        z_event0  = jnp.array(jnp.nan, dtype=jnp.float64)

        def cond_fun(state):
            i, field_kwo, save_ptr, save_buf, done, z_event = state
            return jnp.logical_and(i < steps_total, jnp.logical_not(done))

        def body_fun(state):
            i, field_kwo, save_ptr, save_buf, done, z_event = state

            # ---- one macro-step (sharded) ----
            field_kwo = split_step_sharded(
                field_kwo,
                shard_t=shard_t, replicate=replicate,
                dt=dt, f0=f0, fr=fr, sw=sw,
                deltaZ_NL=deltaZ_NL, gamma=gamma, omega_vec=omega_vec,
                D_half=D_half, Nprop_half=Nprop_half, PML_half=PML_half,
                hrw=hrw, gain_term=gain_term,
                saturation_intensity=saturation_intensity,
                use_gain=use_gain,
                m_nl_substeps=m_nl_substeps
            )

            z_here = (i + 1).astype(jnp.float64) * jnp.asarray(deltaZ_NL, dtype=jnp.float64)

            # decide whether to save this step
            can_save   = save_ptr < save_n
            want_index = jnp.where(can_save, save_idx[save_ptr], -1)
            save_now   = jnp.logical_and(can_save, i == want_index)

            # --- Event gating (PYTHON level) to avoid tracing event_fn when unused ---
            if use_event:
                check_now = ((i + 1) % jnp.asarray(check_every, dtype=i.dtype)) == 0
                need_xyt  = jnp.logical_or(save_now, check_now)
            else:
                check_now = jnp.array(False, dtype=bool)
                need_xyt  = save_now  # no event => only materialize for saving

            # materialize (x,y,t) only if needed
            def compute_xyt(_):
                return _materialize_xyt_from_kwo(field_kwo)
            def dummy_xyt(_):
                return jnp.zeros((Nx, Ny, Nt), dtype=jnp.complex128)

            field_xyt = jax.lax.cond(need_xyt, compute_xyt, dummy_xyt, operand=None)

            # Save snapshot
            def do_save(args):
                save_buf, save_ptr = args
                save_buf = save_buf.at[..., save_ptr].set(field_xyt)
                return save_buf, save_ptr + 1
            def no_save(args):
                return args

            save_buf, save_ptr = jax.lax.cond(save_now, do_save, no_save, (save_buf, save_ptr))

            # Event check (only built when use_event=True)
            if use_event:
                def check_event(_):
                    return event_fn(field_xyt, z_here, payload)  # must return scalar bool
                def skip_event(_):
                    return jnp.array(False)
                triggered = jax.lax.cond(check_now, check_event, skip_event, operand=None)
            else:
                triggered = jnp.array(False)

            done    = jnp.logical_or(done, triggered)
            z_event = jnp.where(jnp.logical_and(triggered, jnp.isnan(z_event)), z_here, z_event)

            return (i + 1, field_kwo, save_ptr, save_buf, done, z_event)

        i_end, field_end_kwo, save_ptr_end, save_buf, done_end, z_event = jax.lax.while_loop(
            cond_fun, body_fun, (i0, field_kwo0, save_ptr0, save_buf0, done0, z_event0)
        )

        # tail fill if requested but not yet written
        def fill_tail(args):
            save_buf, field_end_kwo, save_ptr_end = args
            field_xyt_end = _materialize_xyt_from_kwo(field_end_kwo)
            save_buf = save_buf.at[..., save_ptr_end].set(field_xyt_end)
            return save_buf
        save_buf = jax.lax.cond(save_ptr_end < save_n, fill_tail, lambda x: x[0],
                                (save_buf, field_end_kwo, save_ptr_end))
        
        # If we tail-filled, count is save_ptr_end + 1; else it's save_ptr_end.
        n_saved = jax.lax.select(save_ptr_end < save_n,
                                 save_ptr_end + jnp.int32(1),
                                 save_ptr_end)

        meta = dict(
            steps_executed=i_end,
            stopped_early=done_end,
            z_event=z_event,
            n_saved=n_saved,            
        )
        return save_buf, meta
    
    return _propagate_while_sharded

# --- Sharded versions of your step and scan ----------------------------------

@partial(jax.jit,
         static_argnames=('fr','sw','use_gain','m_nl_substeps',
                          'shard_t','replicate'))
def split_step_sharded(field_kwo, *,
                       shard_t, replicate,   # <- stays in signature
                       dt, f0, fr, sw, deltaZ_NL, gamma, omega_vec,
                       D_half, Nprop_half, PML_half,
                       hrw, gain_term, saturation_intensity, use_gain,
                       m_nl_substeps=1):
    """
    One Strang macro-step with adaptive sharding:
      - keep t-sharded for spatial FFTs,
      - replicate for temporal FFTs + NL,
      - shard back for next spatial FFTs.
    """
    # Hint the layout for spatial FFTs
    field_kwo = jax.lax.with_sharding_constraint(field_kwo, shard_t)

    # 1) half L_k (spectral multiply; local)
    field_kwo = (field_kwo * D_half).astype(jnp.complex128)

    # 2) half L_xy in (x,y,ω) — still t-sharded; 2D IFFT is per-ω batch
    field_xyw = jnp.fft.ifftn(field_kwo, axes=(0,1)).astype(jnp.complex128)
    field_xyw = field_xyw * (PML_half[:, :, None]) * Nprop_half

    # 3) replicate for 1D temporal FFT + NL (avoid distributed 1D FFT)
    field_xyw = jax.lax.with_sharding_constraint(field_xyw, replicate)

    # IFFT_t
    field_xyt = jnp.fft.ifft(field_xyw, axis=2).astype(jnp.complex128)

    # exact Kerr half-kick
    def kerr_half_kick(A, dz_half):
        return (A * jnp.exp(1j * gamma * dz_half * jnp.abs(A)**2)).astype(jnp.complex128)

    field_xyt = kerr_half_kick(field_xyt, 0.5 * deltaZ_NL)

    # residual NL (Heun) for Raman/steepening/gain
    h = deltaZ_NL / jnp.asarray(m_nl_substeps, dtype=jnp.float64)
    def residual_heun(A):
        k1 = _dA_dz_NL_rest(
            A, dt=dt, f0=f0, fr=fr, sw=sw, gamma=gamma,
            omega_vec=omega_vec, hrw=hrw, gain_term=gain_term,
            saturation_intensity=saturation_intensity, use_gain=use_gain
        )
        A1 = A + h * k1
        k2 = _dA_dz_NL_rest(
            A1, dt=dt, f0=f0, fr=fr, sw=sw, gamma=gamma,
            omega_vec=omega_vec, hrw=hrw, gain_term=gain_term,
            saturation_intensity=saturation_intensity, use_gain=use_gain
        )
        return (A + 0.5 * h * (k1 + k2)).astype(jnp.complex128)

    def body(_, A):
        return residual_heun(A)

    field_xyt = jax.lax.fori_loop(0, m_nl_substeps, body, field_xyt)

    # exact Kerr half-kick
    field_xyt = kerr_half_kick(field_xyt, 0.5 * deltaZ_NL)

    # FFT_t
    field_xyw = jnp.fft.fft(field_xyt, axis=2).astype(jnp.complex128)

    # 4) shard back along t for spatial FFTs
    field_xyw = jax.lax.with_sharding_constraint(field_xyw, shard_t)

    # finish L_xy and back to spectral
    field_xyw = field_xyw * (PML_half[:, :, None]) * Nprop_half
    field_kwo = jnp.fft.fftn(field_xyw, axes=(0,1)).astype(jnp.complex128)

    # 5) finish L_k
    field_kwo = (field_kwo * D_half).astype(jnp.complex128)
    return field_kwo

@partial(jax.jit,
         static_argnames=('steps_total','save_n','fr','sw','use_gain',
                          'm_nl_substeps','shard_t','replicate'))
def _propagate_scan_sharded(A0_kwo, *,
                            shard_t, replicate,   # <- stays in signature
                            steps_total, save_idx, save_n,
                            dt, f0, fr, sw, deltaZ_NL, gamma, omega_vec,
                            D_half, PML_half, Nprop_half, hrw, gain_term,
                            saturation_intensity, use_gain, m_nl_substeps):

    """Carries (kx,ky,ω) complex128; saves (x,y,t) complex128 at requested z with proper reshard."""
    Nx, Ny, Nt = A0_kwo.shape
    save_buf0 = jnp.zeros((Nx, Ny, Nt, save_n), dtype=jnp.complex128)
    save_ptr0 = jnp.array(0, dtype=jnp.int32)

    def _save_snapshot(args):
        field_kwo, save_ptr, save_buf, i = args
        # replicate before doing temporal IFFT to materialize full t
        field_kwo = jax.lax.with_sharding_constraint(field_kwo, replicate)
        field_xyw = jnp.fft.ifftn(field_kwo, axes=(0,1)).astype(jnp.complex128)
        field_xyt = jnp.fft.ifft(field_xyw, axis=2).astype(jnp.complex128)
        save_buf = save_buf.at[..., save_ptr].set(field_xyt)
        return (field_kwo, save_ptr + 1, save_buf)

    def _skip_save(args):
        field_kwo, save_ptr, save_buf, i = args
        return (field_kwo, save_ptr, save_buf)

    def body(carry, i):
        field_kwo, save_ptr, save_buf = carry

        field_kwo = split_step_sharded(
            field_kwo,
            shard_t=shard_t, replicate=replicate,
            dt=dt, f0=f0, fr=fr, sw=sw,
            deltaZ_NL=deltaZ_NL, gamma=gamma, omega_vec=omega_vec,
            D_half=D_half, Nprop_half=Nprop_half, PML_half=PML_half,
            hrw=hrw, gain_term=gain_term,
            saturation_intensity=saturation_intensity,
            use_gain=use_gain,
            m_nl_substeps=m_nl_substeps
        )

        can_save   = save_ptr < save_n
        want_index = jnp.where(can_save, save_idx[save_ptr], -1)
        save_now   = jnp.logical_and(can_save, i == want_index)

        field_kwo, save_ptr, save_buf = jax.lax.cond(
            save_now, _save_snapshot, _skip_save, (field_kwo, save_ptr, save_buf, i)
        )

        return (field_kwo, save_ptr, save_buf), None

    # Ensure initial placement along t
    A0_kwo = jax.lax.with_sharding_constraint(A0_kwo, shard_t)

    (field_end_kwo, save_ptr_end, save_buf), _ = jax.lax.scan(
        body,
        (A0_kwo, save_ptr0, save_buf0),
        jnp.arange(steps_total, dtype=jnp.int32)
    )

    def _fill_tail(args):
        save_buf, field_end_kwo, save_ptr_end = args
        field_end_kwo = jax.lax.with_sharding_constraint(field_end_kwo, replicate)
        field_xyw = jnp.fft.ifftn(field_end_kwo, axes=(0,1)).astype(jnp.complex128)
        field_xyt = jnp.fft.ifft(field_xyw, axis=2).astype(jnp.complex128)
        save_buf  = save_buf.at[..., save_ptr_end].set(field_xyt)
        return save_buf

    save_buf = jax.lax.cond(
        save_ptr_end < save_n, _fill_tail, lambda x: x[0],
        (save_buf, field_end_kwo, save_ptr_end)
    )
    return save_buf

# --- Public entry that auto-adapts to device count ---------------------------
def GNLSE3D_propagate(args, A0,
                      *,
                      event_fn=None,
                      event_payload=None,
                      stop_on_event=True,
                      event_check_every: int = 1):
    """
    Public API with early-stopping event.

    Parameters
    ----------
    event_fn : callable or None
        JAX function (field_xyt, z, payload) -> bool. If None or stop_on_event=False, no early stop.
    event_payload : PyTree or None
        Extra data for event_fn (e.g., thresholds, x/y grids).
    stop_on_event : bool
        Enable/disable early stop.
    event_check_every : int
        Evaluate the event every this many macro steps (>=1).
    """
    prep = _prepare_propagation(args, A0)

    # Spectral initial state
    A0_kwo = jnp.fft.fftn(A0.astype(jnp.complex128), axes=(0,1,2)).astype(jnp.complex128)

    # Build mesh/shardings based on Nt and available devices
    _, shard_t, replicate = _make_mesh_for_time_axis(A0_kwo.shape[2])

    # Build the while-loop propagator with event support (closed over sharding & event)
    prop_while = make_propagate_while_sharded(
        shard_t, replicate,
        event_fn=event_fn,
        stop_on_event=stop_on_event,
        event_check_every=event_check_every
    )

    # Warmup (JIT compile)
    _ = prop_while(
        A0_kwo,
        payload=(event_payload if event_payload is not None else {}),
        steps_total=prep["steps_total"],
        save_idx=prep["save_idx"], save_n=prep["save_n"],
        dt=prep["dt"], f0=prep["f0"],
        fr=prep["fr"], sw=prep["sw"],
        deltaZ_NL=prep["deltaZ_NL"],
        gamma=prep["gamma"],
        omega_vec=prep["omega_vec"],
        D_half=prep["D_half"],
        PML_half=prep["PML_half"],
        Nprop_half=prep["Nprop_half"],
        hrw=prep["hrw"],
        gain_term=prep["gain_term"],
        saturation_intensity=args["saturation_intensity"],
        use_gain=prep["use_gain"],
        m_nl_substeps=prep["m_nl_substeps"]
    )

    # Timed run
    t0 = time.time()
    field_saved, meta = prop_while(
        A0_kwo,
        payload=(event_payload if event_payload is not None else {}),
        steps_total=prep["steps_total"],
        save_idx=prep["save_idx"], save_n=prep["save_n"],
        dt=prep["dt"], f0=prep["f0"],
        fr=prep["fr"], sw=prep["sw"],
        deltaZ_NL=prep["deltaZ_NL"],
        gamma=prep["gamma"],
        omega_vec=prep["omega_vec"],
        D_half=prep["D_half"],
        PML_half=prep["PML_half"],
        Nprop_half=prep["Nprop_half"],
        hrw=prep["hrw"],
        gain_term=prep["gain_term"],
        saturation_intensity=args["saturation_intensity"],
        use_gain=prep["use_gain"],
        m_nl_substeps=prep["m_nl_substeps"]
    )
    elapsed = time.time() - t0

    return dict(field=field_saved, dt=prep["dt"], dx=prep["dx"], seconds=elapsed, **meta)

