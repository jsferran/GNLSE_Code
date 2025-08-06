import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def make_xy_t_animation(field4d,
                        z_index=-1,
                        x=None, y=None, t=None,
                        quantity='intensity',      # 'intensity' (|E|^2), 'abs', 'real', 'imag', 'phase'
                        norm='global',             # 'global' or 'per_frame'
                        t_window=None,             # (t_start, t_end) in same units as t (ignored if t is None)
                        frame_window=None,         # (i_start, i_end) inclusive/exclusive frame indices
                        stride=1,                  # take every 'stride' frame in the selected window
                        fps=30,
                        filename='xy_t.gif',
                        dpi=120):
    """
    Animate the transverse (x,y) field vs time at a fixed z, optionally over a narrower time range.

    Parameters
    ----------
    field4d : np.ndarray, complex, shape (Nx, Ny, Nt, Nz)
        The complex field.
    z_index : int
        Which z slice to animate (default last).
    x, y : 1D arrays or None
        Spatial axes (used only for labeling/extent). If None, pixel indices are used.
    t : 1D array or None
        Time axis. Required to use t_window. If None, you can use frame_window instead.
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    norm : str
        'global' -> single color scale over displayed frames;
        'per_frame' -> auto-scale each frame independently.
    t_window : tuple or None
        (t_start, t_end) limits the animation to this physical time window. Requires `t`.
        Both endpoints are inclusive of the nearest sample.
    frame_window : tuple or None
        (i_start, i_end) frame indices. i_end follows Python slicing (exclusive).
        Ignored if t_window is provided.
    stride : int
        Use every `stride`-th frame within the selected window (for decimating long sequences).
    fps : int
        Frames per second for the GIF.
    filename : str
        Output GIF path.
    dpi : int
        Figure DPI for saving.

    Returns
    -------
    filename : str
        Path to the saved GIF.
    """

    assert field4d.ndim == 4, "Expected (Nx, Ny, Nt, Nz)."
    Nx, Ny, Nt, Nz = field4d.shape
    if not (-Nz <= z_index < Nz):
        raise IndexError(f"z_index {z_index} out of range for Nz={Nz}")

    # Extract the (x,y,t) block at fixed z
    F = field4d[..., z_index]  # (Nx, Ny, Nt)

    # Map to requested quantity
    if quantity == 'intensity':
        data_t = np.abs(F)**2
        cbar_label = r'|E|$^2$'
    elif quantity == 'abs':
        data_t = np.abs(F)
        cbar_label = r'|E|'
    elif quantity == 'real':
        data_t = np.real(F)
        cbar_label = 'Re{E}'
    elif quantity == 'imag':
        data_t = np.imag(F)
        cbar_label = 'Im{E}'
    elif quantity == 'phase':
        data_t = np.angle(F)
        cbar_label = 'arg(E)'
    else:
        raise ValueError("quantity must be one of: 'intensity', 'abs', 'real', 'imag', 'phase'")

    # Determine the temporal/frame subset
    if t_window is not None:
        if t is None:
            raise ValueError("t_window was provided but `t` axis is None.")
        t = np.asarray(t)
        if len(t) != Nt:
            raise ValueError("Length of `t` must match Nt dimension of field.")
        t_start, t_end = t_window
        if t_start > t_end:
            t_start, t_end = t_end, t_start
        # nearest-sample indices spanning [t_start, t_end]
        i0 = int(np.clip(np.searchsorted(t, t_start, side='left'), 0, Nt-1))
        i1 = int(np.clip(np.searchsorted(t, t_end,   side='right'), 0, Nt))  # exclusive
    elif frame_window is not None:
        i0, i1 = frame_window
        i0 = int(np.clip(i0, 0, Nt))
        i1 = int(np.clip(i1, i0+1, Nt))  # ensure at least one frame if possible
    else:
        i0, i1 = 0, Nt

    # Apply stride
    frame_indices = np.arange(i0, i1, stride, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("Selected time/frame window is empty after applying stride.")

    data_sel = data_t[..., frame_indices]  # (Nx, Ny, Nf)
    t_sel = t[frame_indices] if (t is not None and len(t) == Nt) else None
    Nf = data_sel.shape[-1]

    # Axes extents for imshow
    if x is not None and y is not None:
        x = np.asarray(x); y = np.asarray(y)
        extent = [x.min(), x.max(), y.min(), y.max()]
        xlabel, ylabel = 'x', 'y'
    else:
        extent = None
        xlabel, ylabel = 'pixel x', 'pixel y'

    # Normalization over the **displayed** frames
    if norm == 'global':
        vmin = np.nanmin(data_sel)
        vmax = np.nanmax(data_sel)
        if vmax == vmin:
            vmax = vmin + (1e-12 if np.isfinite(vmin) else 1.0)
    elif norm == 'per_frame':
        vmin = vmax = None
    else:
        raise ValueError("norm must be 'global' or 'per_frame'")

    # Prepare figure
    fig, ax = plt.subplots()
    im = ax.imshow(data_sel[..., 0].T, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if t_sel is not None:
        title = ax.set_title(f"z index = {z_index}, t = {t_sel[0]}")
    else:
        title = ax.set_title(f"z index = {z_index}, frame = {frame_indices[0]}")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # Frame update
    def update(k):
        frame_data = data_sel[..., k]
        if norm == 'per_frame':
            im.set_clim(np.nanmin(frame_data), np.nanmax(frame_data))
        im.set_data(frame_data.T)
        if t_sel is not None:
            title.set_text(f"z index = {z_index}, t = {t_sel[k]}")
        else:
            title.set_text(f"z index = {z_index}, frame = {frame_indices[k]}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=Nf, interval=1000.0/fps, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    return filename

import numpy as np

def power_vs_time_from_results(results, z_index=-1, dx=None, dy=None):
    """
    Compute temporal power P(t) at the fiber output (or any z_index).

    Parameters
    ----------
    results : dict
        Must contain 'fields' or 'field' with shape (Nx, Ny, Nt, Nz).
        If dx, dy are not provided, results should also contain 1D 'x' and 'y'.
    z_index : int
        Which z slice to use; default -1 (output).
    dx, dy : float or None
        Spatial sample spacings. If None, inferred from results['x'], results['y'].

    Returns
    -------
    P_t : np.ndarray, shape (Nt,)
        Temporal power at the chosen z, i.e. sum_{x,y} |E(x,y,t,z)|^2 * dx * dy.
    """
    # get field
    field4d = results.get("fields", results.get("field", None))
    if field4d is None:
        raise KeyError("results must contain 'fields' or 'field' with shape (Nx, Ny, Nt, Nz).")
    if field4d.ndim != 4:
        raise ValueError(f"Expected field shape (Nx, Ny, Nt, Nz); got {field4d.shape}.")

    Nx, Ny, Nt, Nz = field4d.shape
    if not (-Nz <= z_index < Nz):
        raise IndexError(f"z_index {z_index} out of range for Nz={Nz}.")

    # spacings
    if dx is None or dy is None:
        x = results.get("x", None)
        y = results.get("y", None)
        if x is None or y is None:
            raise ValueError("Provide dx, dy, or include 1D arrays 'x' and 'y' in results to infer them.")
        x = np.asarray(x); y = np.asarray(y)
        if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
            raise ValueError("results['x'] and results['y'] must be 1D arrays of length >= 2.")
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])

    # slice at z and integrate over x,y
    F = field4d[..., z_index]                # (Nx, Ny, Nt)
    P_t = np.sum(np.abs(F)**2, axis=(0, 1))  # integrate over x,y (no spacings yet)
    P_t = P_t * dx * dy                      # apply area element

    # ensure real (tiny imag can appear from numerics)
    return np.real_if_close(P_t, tol=1000)



def make_xy_z_animation(field4d,
                        t_index=-1,
                        x=None, y=None, z=None,
                        quantity='intensity',      # 'intensity' (|E|^2), 'abs', 'real', 'imag', 'phase'
                        norm='global',             # 'global' or 'per_frame'
                        z_window=None,             # (z_start, z_end) in same units as z (ignored if z is None)
                        frame_window=None,         # (i_start, i_end) inclusive/exclusive frame indices
                        stride=1,                  # take every 'stride' frame in the selected window
                        fps=30,
                        filename='xy_z.gif',
                        dpi=120):
    """
    Animate the transverse (x,y) field vs propagation distance z at a fixed time index t.

    Parameters
    ----------
    field4d : np.ndarray, complex, shape (Nx, Ny, Nt, Nz)
        The complex field.
    t_index : int
        Which time slice to animate (default last).
    x, y : 1D arrays or None
        Spatial axes (used only for labeling/extent). If None, pixel indices are used.
    z : 1D array or None
        z axis. Required to use z_window. If None, you can use frame_window instead.
    quantity : str
        'intensity', 'abs', 'real', 'imag', or 'phase'.
    norm : str
        'global' -> single color scale over displayed frames;
        'per_frame' -> auto-scale each frame independently.
    z_window : tuple or None
        (z_start, z_end) limits the animation to this physical z window. Requires `z`.
        Both endpoints are inclusive of the nearest sample.
    frame_window : tuple or None
        (i_start, i_end) frame indices along z. i_end follows Python slicing (exclusive).
        Ignored if z_window is provided.
    stride : int
        Use every `stride`-th frame within the selected window (for decimating long sequences).
    fps : int
        Frames per second for the GIF.
    filename : str
        Output GIF path.
    dpi : int
        Figure DPI for saving.

    Returns
    -------
    filename : str
        Path to the saved GIF.
    """

    assert field4d.ndim == 4, "Expected (Nx, Ny, Nt, Nz)."
    Nx, Ny, Nt, Nz = field4d.shape
    if not (-Nt <= t_index < Nt):
        raise IndexError(f"t_index {t_index} out of range for Nt={Nt}")

    # Extract the (x,y,z) block at fixed t
    F = field4d[:, :, t_index, :]  # (Nx, Ny, Nz)

    # Map to requested quantity
    if quantity == 'intensity':
        data_z = np.abs(F)**2
        cbar_label = r'|E|$^2$'
    elif quantity == 'abs':
        data_z = np.abs(F)
        cbar_label = r'|E|'
    elif quantity == 'real':
        data_z = np.real(F)
        cbar_label = 'Re{E}'
    elif quantity == 'imag':
        data_z = np.imag(F)
        cbar_label = 'Im{E}'
    elif quantity == 'phase':
        data_z = np.angle(F)
        cbar_label = 'arg(E)'
    else:
        raise ValueError("quantity must be one of: 'intensity', 'abs', 'real', 'imag', 'phase'")

    # Determine the z/frame subset
    if z_window is not None:
        if z is None:
            raise ValueError("z_window was provided but `z` axis is None.")
        z = np.asarray(z)
        if len(z) != Nz:
            raise ValueError("Length of `z` must match Nz dimension of field.")
        z_start, z_end = z_window
        if z_start > z_end:
            z_start, z_end = z_end, z_start
        i0 = int(np.clip(np.searchsorted(z, z_start, side='left'), 0, Nz-1))
        i1 = int(np.clip(np.searchsorted(z, z_end,   side='right'), 0, Nz))  # exclusive
    elif frame_window is not None:
        i0, i1 = frame_window
        i0 = int(np.clip(i0, 0, Nz))
        i1 = int(np.clip(i1, i0+1, Nz))  # ensure at least one frame if possible
    else:
        i0, i1 = 0, Nz

    # Apply stride
    frame_indices = np.arange(i0, i1, stride, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("Selected z/frame window is empty after applying stride.")

    data_sel = data_z[..., frame_indices]              # (Nx, Ny, Nf)
    z_sel = z[frame_indices] if (z is not None and len(z) == Nz) else None
    Nf = data_sel.shape[-1]

    # Axes extents for imshow
    if x is not None and y is not None:
        x = np.asarray(x); y = np.asarray(y)
        extent = [x.min(), x.max(), y.min(), y.max()]
        xlabel, ylabel = 'x', 'y'
    else:
        extent = None
        xlabel, ylabel = 'pixel x', 'pixel y'

    # Normalization over the displayed frames
    if norm == 'global':
        vmin = np.nanmin(data_sel)
        vmax = np.nanmax(data_sel)
        if vmax == vmin:
            vmax = vmin + (1e-12 if np.isfinite(vmin) else 1.0)
    elif norm == 'per_frame':
        vmin = vmax = None
    else:
        raise ValueError("norm must be 'global' or 'per_frame'")

    # Prepare figure
    fig, ax = plt.subplots()
    im = ax.imshow(data_sel[..., 0].T, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if z_sel is not None:
        title = ax.set_title(f"t index = {t_index}, z = {z_sel[0]}")
    else:
        title = ax.set_title(f"t index = {t_index}, frame = {frame_indices[0]}")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # Frame update
    def update(k):
        frame_data = data_sel[..., k]
        if norm == 'per_frame':
            im.set_clim(np.nanmin(frame_data), np.nanmax(frame_data))
        im.set_data(frame_data.T)
        if z_sel is not None:
            title.set_text(f"t index = {t_index}, z = {z_sel[k]}")
        else:
            title.set_text(f"t index = {t_index}, frame = {frame_indices[k]}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=Nf, interval=1000.0/fps, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    return filename
