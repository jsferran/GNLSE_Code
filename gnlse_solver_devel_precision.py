

import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# IMPORTANT: Do NOT force x64 globally. Choose via precision="fp32"/"fp64" at call-time.
# If you insist on a default, uncomment the next line, but fp32 is far lighter:
# jax.config.update("jax_enable_x64", False)

# --------------------------------------------------------------------------------------
# Precision policy (controls both JAX and NumPy dtypes used throughout)
# --------------------------------------------------------------------------------------
def _resolve_precision(precision: str | None):
    p = (precision or "fp64").lower()
    if p in ("fp32", "32", "single"):
        return jnp.float32, jnp.complex64, np.float32
    elif p in ("fp64", "64", "double"):
        return jnp.float64, jnp.complex128, np.float64
    else:
        raise ValueError("precision must be 'fp32' or 'fp64'")

C0 = 299_792_458.0
LN10_OVER_10 = np.log(10)/10


# --------------------------------------------------------------------------------------
# Raman kernel (fft of causal response) — dtype-aware
# --------------------------------------------------------------------------------------
def _make_hrw(Nt: int, dt: float, t1=12.2e-15, t2=32.0e-15, *, real_dtype=jnp.float64):
    complex_dtype = jnp.complex64 if real_dtype == jnp.float32 else jnp.complex128
    t = (dt * jnp.arange(Nt, dtype=real_dtype))
    h = ((t1**2 + t2**2) / (t1 * t2**2)) * jnp.exp(-t/real_dtype(t2)) * jnp.sin(t/real_dtype(t1))
    H = jnp.fft.fft(h).astype(complex_dtype) * real_dtype(dt)  # critical dt factor
    return H

# --------------------------------------------------------------------------------------
# Residual NL (Raman + self-steepening + gain/saturation)
# --------------------------------------------------------------------------------------
def _dA_dz_NL_rest(A_xy_t,
                   *, dt: float, f0: float, fr: float, sw: int, gamma: float,
                   omega_vec: jnp.ndarray, hrw: jnp.ndarray, gain_term: jnp.ndarray,
                   saturation_intensity: float, use_gain: bool) -> jnp.ndarray:
    CD = A_xy_t.dtype
    RD = jnp.float32 if CD == jnp.complex64 else jnp.float64
    ONEJ = jax.lax.complex(RD(0.0), RD(1.0))

    absA2 = jnp.abs(A_xy_t)**2
    NL = jnp.zeros_like(A_xy_t, dtype=CD)

    # Raman
    if fr != 0.0:
        Iw = jnp.fft.fft(absA2, axis=2).astype(CD)
        Hw = hrw[None, None, :].astype(CD)
        Raman = jnp.fft.ifft(Hw * Iw, axis=2).astype(CD)
        Raman = jnp.nan_to_num(Raman, nan=0.0, posinf=0.0, neginf=0.0).astype(CD)
        NL = NL + (fr * Raman).astype(CD) * A_xy_t

    # Self-steepening
    if sw == 1:
        NLw = jnp.fft.fft(NL, axis=2).astype(CD)
        NLw = NLw * (RD(1.0) + omega_vec[None, None, :] / (RD(2.0) * jnp.pi * RD(f0)))
        NL = jnp.fft.ifft(NLw, axis=2).astype(CD)

    dA = (ONEJ * RD(gamma)).astype(CD) * NL

    # Saturable gain (optional)
    def _add_gain(args):
        dA_in, A_in, absA2_in = args
        power_xy  = jnp.sum(absA2_in, axis=2) * RD(dt)
        gain_pref = RD(1.0) / (RD(1.0) + power_xy / RD(saturation_intensity))
        gain_pref = gain_pref[:, :, None].astype(RD)
        Aw = jnp.fft.fft(A_in, axis=2).astype(CD)
        A_gain_w = gain_term.astype(CD) * (gain_pref.astype(CD) * Aw)
        return dA_in + jnp.fft.ifft(A_gain_w, axis=2).astype(CD)

    dA = jax.lax.cond(use_gain, _add_gain, lambda args: dA, (dA, A_xy_t, absA2))
    return dA.astype(CD)

# --------------------------------------------------------------------------------------
# Precompute propagators (dtype-aware, memory-lean)
# --------------------------------------------------------------------------------------
def _prepare_propagation(args, A0, *, precision: str = "fp64"):
    RD, CD, NPD = _resolve_precision(precision)

    # Grid
    Lx, Ly, Lz, Lt = args["Lx"], args["Ly"], args["Lz"], args["Lt"]
    Nx, Ny, Nt     = args["Nx"], args["Ny"], args["Nt"]
    dx, dy, dt     = Lx/Nx, Ly/Ny, Lt/Nt

    deltaZ_linear = float(args["deltaZ"])
    deltaZ_NL     = float(args["deltaZ_NL"])
    steps_total   = int(round(Lz / deltaZ_linear))



    # Save indices
    save_at_m = np.asarray(args["save_at"], dtype=np.float64)
    save_idx  = np.unique(np.clip(np.rint(save_at_m / deltaZ_linear).astype(np.int32),
                                  0, max(0, steps_total-1)))
    save_idx  = jnp.asarray(save_idx, dtype=jnp.int32)
    save_n    = int(save_idx.size)

    # Physics constants
    c0      = 2.997_924_58e8
    lambda0 = float(args["lambda0"])
    f0      = c0 / lambda0
    omega0  = RD(2) * jnp.pi * RD(f0)
    n2      = float(args["n2"])
    gamma   = RD(n2) * omega0 / RD(c0)

    beta0, beta1, beta2 = float(args["beta0"]), float(args["beta1"]), float(args["beta2"])
    gain_coeff, gain_fwhm = float(args["gain_coeff"]), float(args["gain_fwhm"])
    use_gain = bool(gain_coeff != 0.0)
    t1, t2 = float(args["t1"]), float(args["t2"])

    # k / ω grids
    omega = RD(2) * jnp.pi * jnp.fft.fftfreq(Nt, RD(dt)).astype(RD)
    kx    = RD(2) * jnp.pi * jnp.fft.fftfreq(Nx, RD(dx)).astype(RD)
    ky    = RD(2) * jnp.pi * jnp.fft.fftfreq(Ny, RD(dy)).astype(RD)

    KX, KY, OMEGA = kx[:, None, None], ky[None, :, None], omega[None, None, :]

    # Material index (x,y,ω)
    n_xyomega = jnp.asarray(args["n_xyomega"], dtype=RD)
    n_xyomega = jnp.broadcast_to(n_xyomega, (Nx, Ny, Nt))
    n_eff_omega = n_xyomega[Nx//2, Ny//2, :]
    beta_eff = (n_eff_omega[None,None,:] * (omega0 + OMEGA) / RD(c0)).astype(RD)

    # Linear spectral generator and halves
    ONEJ = jax.lax.complex(RD(0.0), RD(1.0))
    rad = beta_eff**2 - KX**2 - KY**2
    sqrt_term = jnp.sqrt(rad.astype(CD))                  # complex sqrt
    D = ONEJ *( sqrt_term - RD(beta0) - RD(beta1)*OMEGA - RD(0.5)*RD(beta2)*OMEGA**2).astype(CD)
    D_half = jnp.exp(D * RD(deltaZ_linear/2)).astype(CD)

    # PML (x,y) and waveguide phase
    pml_thickness = int(args["pml_thickness"])
    pml_Wmax      = float(args["pml_Wmax"])

    idx = jnp.arange(Nx); idy = jnp.arange(Ny)
    d_x = jnp.minimum(idx, (Nx-1)-idx)
    d_y = jnp.minimum(idy, (Ny-1)-idy)
    ramp_x = jnp.where(d_x < pml_thickness, RD(pml_Wmax)*((RD(pml_thickness)-d_x)/RD(pml_thickness))**2, RD(0))
    ramp_y = jnp.where(d_y < pml_thickness, RD(pml_Wmax)*((RD(pml_thickness)-d_y)/RD(pml_thickness))**2, RD(0))
    W2d = (ramp_x[:,None] + ramp_y[None,:]).astype(RD)
    PML_half = jnp.exp(-W2d * RD(deltaZ_linear/2)).astype(RD)

    Nprop_half = jnp.exp(
        (ONEJ * beta_eff / RD(2)) * ((n_xyomega/n_eff_omega[None,None,:])**2 - RD(1.0)) * RD(deltaZ_linear/2)
    ).astype(CD)

    # Raman kernel
    hrw = _make_hrw(Nt, dt, t1, t2, real_dtype=RD)

    # Gain spectral envelope
    if use_gain:
        omega_fwhm = RD(2) * jnp.pi * RD(gain_fwhm)
        omega_mid  = omega_fwhm / (RD(2) * jnp.sqrt(jnp.log(RD(2))))
        g0 = RD(gain_coeff)/RD(2)
        gain_term = (g0 * jnp.exp(-(OMEGA**2)/(RD(2)*omega_mid**2))).astype(CD)
        use_gain_flag = True
    else:
        gain_term = jnp.zeros((1,1,Nt), dtype=CD)
        use_gain_flag = False

    # NL stepping strategy
    if deltaZ_NL <= deltaZ_linear:
        nl_outer_subcycles = int(max(1, round(deltaZ_linear / deltaZ_NL)))
        skip_nl_every = 1
    else:
        nl_outer_subcycles = 1
        skip_nl_every = int(max(1, round(deltaZ_NL / deltaZ_linear)))

    m_nl = int(args.get("m_nl_substeps", 1))

    return dict(
        steps_total=int(steps_total),
        save_idx=save_idx, save_n=int(save_n),
        dt=RD(dt), dx=RD(dx), dy=RD(dy),
        omega_vec=omega.astype(RD),
        f0=RD(C0/float(args["lambda0"])),
        gamma=RD(gamma),
        D_half=D_half, PML_half=PML_half, Nprop_half=Nprop_half,
        hrw=hrw, gain_term=gain_term,
        fr=float(args["fr"]), sw=int(args["sw"]),
        deltaZ_linear=RD(deltaZ_linear),
        deltaZ_NL=RD(deltaZ_NL),
        use_gain=use_gain_flag,
        m_nl_substeps=m_nl,
        nl_outer_subcycles=nl_outer_subcycles,
        skip_nl_every=skip_nl_every,
    )

# --------------------------------------------------------------------------------------
# Sharding utilities
# --------------------------------------------------------------------------------------
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def _best_1d_factor(n_devices: int, n_axis: int) -> int:
    best = 1
    for d in range(1, n_devices + 1):
        if (n_devices % d == 0) and (n_axis % d == 0):
            best = d
    return best

def _make_mesh_for_time_axis(Nt: int):
    devs = jax.devices()
    ndev = len(devs)
    if ndev == 0:
        raise RuntimeError("No JAX devices available.")
    n_shards_t = _best_1d_factor(ndev, Nt)
    mesh_arr = np.array(devs[:max(1, n_shards_t)]).reshape((max(1, n_shards_t),))
    mesh = Mesh(mesh_arr, axis_names=('t',))
    shard_t   = NamedSharding(mesh, P(None, None, 't'))   # shard along Nt (x,y,t)
    replicate = NamedSharding(mesh, P(None, None, None))  # fully replicated
    return mesh, shard_t, replicate

# --------------------------------------------------------------------------------------
# Scan-based propagator with memory-minimizing on-demand materialization
# --------------------------------------------------------------------------------------
def make_propagate_scan_sharded_checkpointed(
    shard_t, replicate,
    *,
    event_fn=None,
    stop_on_event=True,
    event_check_every: int = 1,
    # checkpointing knobs
    strategy: str = "segments",
    segment_len: int = 16,
    tree_depth: int = 2,
    base_len: int = 32,
):
    use_event   = bool(callable(event_fn) and stop_on_event)
    check_every = int(max(1, event_check_every))

    def _materialize_xyt_from_kwo(field_kwo):
        field_kwo = jax.lax.with_sharding_constraint(field_kwo, replicate)
        field_xyw = jnp.fft.ifftn(field_kwo, axes=(0,1))
        field_xyt = jnp.fft.ifft(field_xyw, axis=2)
        return field_xyt

    @partial(
        jax.jit,
        static_argnames=('steps_total','save_n','fr','sw','use_gain',
                         'm_nl_substeps','nl_outer_subcycles','skip_nl_every',
                         'strategy','segment_len','tree_depth','base_len',
                         'save_as_fp32'),
        donate_argnums=(0,),
    )
    def _propagate_scan_ckpt(
        A0_kwo: jnp.ndarray,
        *,
        payload,
        steps_total: int,
        save_idx: jnp.ndarray, save_n: int,
        dt: float, f0: float,
        fr: float, sw: int,
        deltaZ_linear: float,
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
        m_nl_substeps: int,
        nl_outer_subcycles: int,
        skip_nl_every: int,
        strategy: str = "segments",
        segment_len: int = 16,
        tree_depth: int = 2,
        base_len: int = 32,
        save_as_fp32: bool = False,
    ):
        CD_sim = A0_kwo.dtype
        CD_save = (jnp.complex64 if (save_as_fp32 and CD_sim == jnp.complex128) else CD_sim)

        Nx, Ny, Nt = A0_kwo.shape
        field_kwo0 = jax.lax.with_sharding_constraint(A0_kwo, shard_t)

        # minimal save buffer
        save_buf0  = jnp.zeros((Nx, Ny, Nt, save_n), dtype=CD_save) if save_n > 0 else jnp.zeros((Nx,Ny,Nt,0), dtype=CD_save)
        save_ptr0  = jnp.array(0, dtype=jnp.int32)
        i0         = jnp.array(0, dtype=jnp.int32)
        done0      = jnp.array(False)
        z_event0   = jnp.array(jnp.nan, dtype=(jnp.float32 if CD_sim == jnp.complex64 else jnp.float64))

        # one step
        def one_step(state, _):
            i, field_kwo, save_ptr, save_buf, done, z_event = state
            apply_nl = ((i + 1) % jnp.asarray(skip_nl_every, dtype=i.dtype)) == 0

            field_kwo = split_step_sharded(
                field_kwo,
                shard_t=shard_t, replicate=replicate,
                dt=dt, f0=f0, fr=fr, sw=sw,
                deltaZ_linear=deltaZ_linear, deltaZ_NL=deltaZ_NL,
                gamma=gamma, omega_vec=omega_vec,
                D_half=D_half, Nprop_half=Nprop_half, PML_half=PML_half,
                hrw=hrw, gain_term=gain_term,
                saturation_intensity=saturation_intensity,
                use_gain=use_gain,
                m_nl_substeps=m_nl_substeps,
                nl_outer_subcycles=nl_outer_subcycles,
                apply_nl=apply_nl,
            )

            z_here = (i + 1).astype(z_event.dtype) * deltaZ_linear

            # ---- save path
            can_save   = save_ptr < save_n
            want_index = jnp.where(can_save, save_idx[save_ptr], -1)
            save_now   = jnp.logical_and(can_save, i == want_index)

            def do_save(args):
                field_kwo_in, save_buf_in, save_ptr_in = args
                xyt = _materialize_xyt_from_kwo(field_kwo_in).astype(CD_save)
                save_buf_out = save_buf_in.at[..., save_ptr_in].set(xyt)
                return (save_buf_out, save_ptr_in + 1)

            save_buf, save_ptr = jax.lax.cond(
                save_now,
                do_save,
                lambda args: (args[1], args[2]),
                (field_kwo, save_buf, save_ptr),
            )

            # ---- event path
            if use_event:
                check_now = ((i + 1) % jnp.asarray(check_every, dtype=i.dtype)) == 0

                def do_event(args):
                    field_kwo_in, z_val = args
                    xyt = _materialize_xyt_from_kwo(field_kwo_in)
                    return event_fn(xyt, z_val, payload)

                triggered = jax.lax.cond(
                    check_now,
                    do_event,
                    lambda _: jnp.array(False),
                    (field_kwo, z_here),
                )
            else:
                triggered = jnp.array(False)

            done    = jnp.logical_or(done, triggered)
            z_event = jnp.where(jnp.logical_and(triggered, jnp.isnan(z_event)), z_here, z_event)

            return (i + 1, field_kwo, save_ptr, save_buf, done, z_event), None

        # runners
        def run_segment(state, n: int, remat: bool):
            body = one_step
            if remat:
                body = jax.checkpoint(body, prevent_cse=False)
            state, _ = jax.lax.scan(body, state, xs=None, length=int(n))
            return state

        def run_none(state):
            return run_segment(state, int(steps_total), remat=False)

        def run_segments(state):
            N = int(steps_total); S = int(max(1, segment_len)); k = 0
            while k < N:
                nseg = min(S, N - k)
                state = run_segment(state, nseg, remat=True)
                k += nseg
            return state

        def run_tree(state):
            N = int(steps_total); D = int(tree_depth); B = int(base_len)
            def build(n, d):
                if (n <= B) or (d <= 0): return [n]
                n1 = n // 2; n2 = n - n1
                return build(n1, d-1) + build(n2, d-1)
            for nseg in build(N, D):
                state = run_segment(state, int(nseg), remat=True)
            return state

        state0 = (i0, field_kwo0, save_ptr0, save_buf0, done0, z_event0)
        if strategy == "none":
            state_end = run_none(state0)
        elif strategy == "segments":
            state_end = run_segments(state0)
        elif strategy == "tree":
            state_end = run_tree(state0)
        else:
            raise ValueError("strategy must be one of: 'none', 'segments', 'tree'")

        i_end, field_end_kwo, save_ptr_end, save_buf, done_end, z_event = state_end

        # Tail save (if last requested frame not yet saved)
        def _tail_save(args):
            save_buf_in, field_end_kwo_in, save_ptr_in = args
            xyt_end = _materialize_xyt_from_kwo(field_end_kwo_in).astype(CD_save)
            return save_buf_in.at[..., save_ptr_in].set(xyt_end)

        pred_tail = save_ptr_end < jnp.int32(save_n)
        save_buf = jax.lax.cond(
            pred_tail,
            _tail_save,
            lambda x: x[0],
            (save_buf, field_end_kwo, save_ptr_end),
        )

        n_saved = jax.lax.select(pred_tail, save_ptr_end + jnp.int32(1), save_ptr_end)

        meta = dict(
            steps_executed=i_end,
            stopped_early=done_end,
            z_event=z_event,
            n_saved=n_saved,
        )
        return save_buf, meta

    return _propagate_scan_ckpt

# --------------------------------------------------------------------------------------
# Split-step kernel (donates carry; real masks, complex field)
# --------------------------------------------------------------------------------------
@partial(
    jax.jit,
    static_argnames=('fr','sw','use_gain','m_nl_substeps','nl_outer_subcycles','shard_t','replicate'),
    donate_argnums=(0,)
)
def split_step_sharded(field_kwo, *,
                       shard_t, replicate,
                       dt, f0, fr, sw,
                       deltaZ_linear, deltaZ_NL,
                       gamma, omega_vec,
                       D_half, Nprop_half, PML_half,
                       hrw, gain_term, saturation_intensity, use_gain,
                       m_nl_substeps=1,
                       nl_outer_subcycles=1,
                       apply_nl=True,
                       ):

    CD = field_kwo.dtype
    RD = jnp.float32 if CD == jnp.complex64 else jnp.float64
    ONEJ = jax.lax.complex(RD(0.0), RD(1.0))

    

    # 1) half L_k + masks
    field_kwo = jax.lax.with_sharding_constraint(field_kwo, shard_t)
    field_kwo = field_kwo * D_half
    

    # 2) half L_xy
    field_xyw = jnp.fft.ifftn(field_kwo, axes=(0,1))
    field_xyw = field_xyw * (PML_half[:, :, None].astype(RD)) * Nprop_half

    def _do_nl(field_xyw_local, apply_residual: bool):
        field_xyw_rep = jax.lax.with_sharding_constraint(field_xyw_local, replicate)
        field_xyt = jnp.fft.ifft(field_xyw_rep, axis=2)

        def kerr_half(A, dz):
            return A * jnp.exp(ONEJ * RD(gamma) * RD(0.5) * RD(dz) * jnp.abs(A)**2)

        def residual_heun(A, dz):
            h = RD(dz) / RD(m_nl_substeps)
            def step(a,_):
                k1 = _dA_dz_NL_rest(a, dt=dt, f0=f0, fr=fr, sw=sw, gamma=gamma,
                                    omega_vec=omega_vec, hrw=hrw, gain_term=gain_term,
                                    saturation_intensity=saturation_intensity, use_gain=use_gain)
                a1 = a + h*k1
                k2 = _dA_dz_NL_rest(a1, dt=dt, f0=f0, fr=fr, sw=sw, gamma=gamma,
                                    omega_vec=omega_vec, hrw=hrw, gain_term=gain_term,
                                    saturation_intensity=saturation_intensity, use_gain=use_gain)
                return a + RD(0.5)*h*(k1+k2), None
            A_out, _ = jax.lax.scan(step, A, xs=None, length=m_nl_substeps)
            return A_out

        m  = int(nl_outer_subcycles)
        dz = deltaZ_linear / m

        def one_cycle(A,_):
            A = kerr_half(A, dz)                                # ← always
            A = jax.lax.cond(apply_residual,
                            lambda a: residual_heun(a, dz),
                            lambda a: a,
                            A)
            A = kerr_half(A, dz)                                # ← always
            return A, None

        field_xyt, _ = jax.lax.scan(one_cycle, field_xyt, xs=None, length=m)
        return jnp.fft.fft(field_xyt, axis=2)


    field_xyw = _do_nl(field_xyw, apply_residual=apply_nl)
    field_xyw = jax.lax.with_sharding_constraint(field_xyw, shard_t)

    # finish L_xy and go back to spectral
    field_xyw = field_xyw * (PML_half[:, :, None].astype(RD)) * Nprop_half
    field_kwo = jnp.fft.fftn(field_xyw, axes=(0,1))

    # 5) finish L_k + masks
    field_kwo = field_kwo * D_half
    return field_kwo

# --------------------------------------------------------------------------------------
# Public entry (adds save compression + precision knob)
# --------------------------------------------------------------------------------------
def GNLSE3D_propagate(
    args, A0,
    *,
    event_fn=None,
    event_payload=None,
    stop_on_event=True,
    event_check_every: int = 1e10,
    ckpt_strategy: str | None = None,
    ckpt_segment_len: int | None = None,
    ckpt_tree_depth: int | None = None,
    ckpt_base_len: int | None = None,
    precision: str | None = None,      # "fp32" or "fp64" (default fp64)
    save_as_fp32: bool = True,         # NEW: compress snapshots when possible
):
    # Decide precision (arg overrides args["precision"])
    prec = precision or args.get("precision", "fp64")
    RD, CD, _ = _resolve_precision(prec)

    prep = _prepare_propagation(args, A0, precision=prec)

    # Cast launch field once to simulation dtype
    A0_kwo = jnp.fft.fftn(jnp.asarray(A0, dtype=CD), axes=(0,1,2))

    _, shard_t, replicate = _make_mesh_for_time_axis(A0_kwo.shape[2])

    strategy  = ckpt_strategy   or args.get("ckpt_strategy",   "segments")
    seglen    = ckpt_segment_len if ckpt_segment_len is not None else args.get("ckpt_segment_len", 16)
    treedepth = ckpt_tree_depth  if ckpt_tree_depth  is not None else args.get("ckpt_tree_depth", 2)
    baselen   = ckpt_base_len    if ckpt_base_len    is not None else args.get("ckpt_base_len", 32)

    prop_scan = make_propagate_scan_sharded_checkpointed(
        shard_t, replicate,
        event_fn=event_fn,
        stop_on_event=stop_on_event,
        event_check_every=event_check_every,
        strategy=strategy,
        segment_len=int(seglen),
        tree_depth=int(treedepth),
        base_len=int(baselen),
    )


        # Timed run
    t0 = time.time()
    field_saved, meta = prop_scan(
        A0_kwo,
        payload=(event_payload if event_payload is not None else {}),
        steps_total=prep["steps_total"],
        save_idx=prep["save_idx"], save_n=prep["save_n"],
        dt=prep["dt"], f0=prep["f0"],
        fr=prep["fr"], sw=prep["sw"],
        deltaZ_linear=prep["deltaZ_linear"],
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
        m_nl_substeps=prep["m_nl_substeps"],
        nl_outer_subcycles=prep["nl_outer_subcycles"],
        skip_nl_every=prep["skip_nl_every"],
        strategy=strategy, segment_len=int(seglen), tree_depth=int(treedepth), base_len=int(baselen),
        save_as_fp32=bool(save_as_fp32),
    )
    elapsed = time.time() - t0

    return dict(field=field_saved, dt=prep["dt"], dx=prep["dx"], seconds=elapsed, **meta)
