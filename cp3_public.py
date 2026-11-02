"""CP3 public utilities: conditions, FEA, compliance, a reference SIMP, and file I/O.

Ships to students. The manufacturing and service rules live in cp3_rules.py, which does not ship.

Conventions
-----------
- A design is an (80, 80) array indexed design[y, x]; row 0 is the top edge. Values > 0.5 are solid.
- Positions are in mm (1 px = 1 mm) on the 81 x 81 grid of element corners, x to the right and y down,
  matching the array indexing.
- A condition is 8 numbers (see COND_COLS):
    load_x, load_y   centre of the load disc (radius LOAD_R); the load is spread over the disc
    force_x, force_y unit load direction, in the same x-right / y-down frame
    vf_cap           volume-fraction cap
    sup_1..sup_3     centre of each clamped support segment (SEG_LEN mm long), as distance along the plate's
                     boundary, clockwise from the top-left corner: top edge 0-80, right 80-160, bottom 160-240,
                     left 240-320
"""
import os
import re

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

N = 80
PERIM = 4*N
SEG_LEN = 8
LOAD_R = 3.0
NU = 0.3
EMIN = 1e-6
COND_COLS = ['load_x', 'load_y', 'force_x', 'force_y', 'vf_cap', 'sup_1', 'sup_2', 'sup_3']


# ---------------------------------------------------------------- mesh (top88: node (ix, iy) -> ix*(N+1) + iy, y dof points up)
def _ke(nu=NU):
    k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    idx = [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 7, 6, 5, 4, 3, 2], [2, 7, 0, 5, 6, 3, 4, 1], [3, 6, 5, 0, 7, 2, 1, 4],
           [4, 5, 6, 7, 0, 1, 2, 3], [5, 4, 3, 2, 1, 0, 7, 6], [6, 3, 4, 1, 2, 7, 0, 5], [7, 2, 1, 4, 3, 6, 5, 0]]
    return k[np.array(idx)]/(1 - nu**2)


KE = _ke()
NDOF = 2*(N + 1)**2
_elx, _ely = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
_n1 = ((N + 1)*_elx + _ely).ravel()                 # element el = ely + elx*N
_n2 = ((N + 1)*(_elx + 1) + _ely).ravel()
EDOF = np.stack([2*_n1 + 2, 2*_n1 + 3, 2*_n2 + 2, 2*_n2 + 3, 2*_n2, 2*_n2 + 1, 2*_n1, 2*_n1 + 1], 1)
_IK = np.repeat(EDOF, 8, axis=1).ravel()
_JK = np.tile(EDOF, (1, 8)).ravel()


def node_id(ix, iy):
    return np.asarray(ix)*(N + 1) + np.asarray(iy)


def img2vec(img):
    return np.asarray(img, dtype=float).T.ravel()


def vec2img(v):
    return np.asarray(v).reshape(N, N).T


# ---------------------------------------------------------------- conditions
def perimeter_node(p):
    """Integer boundary position (clockwise from the top-left corner) -> node (ix, iy)."""
    p = int(p) % PERIM
    if p < N:
        return p, 0
    if p < 2*N:
        return N, p - N
    if p < 3*N:
        return 3*N - p, N
    return 0, PERIM - p


def support_nodes(cond):
    """Nodes clamped by the three support segments, as a list of (ix, iy)."""
    out = []
    for t in cond[5:8]:
        start = int(round(t - SEG_LEN/2))
        out += [perimeter_node(p) for p in range(start, start + SEG_LEN + 1)]
    return sorted(set(out))


def support_elements(cond):
    """Elements touching a clamped node (the material that has to reach a support)."""
    m = np.zeros((N, N), bool)
    for ix, iy in support_nodes(cond):
        m[max(iy - 1, 0):iy + 1, max(ix - 1, 0):ix + 1] = True
    return m


def load_nodes(cond):
    """Nodes inside the load disc."""
    ix, iy = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing='ij')
    m = (ix - cond[0])**2 + (iy - cond[1])**2 <= LOAD_R**2
    return list(zip(ix[m].tolist(), iy[m].tolist()))


def touching(nodes, design):
    """Subset of nodes that touch at least one solid element of a binary design."""
    s = np.pad(np.asarray(design, bool), 1)
    return [(ix, iy) for ix, iy in nodes if s[iy:iy + 2, ix:ix + 2].any()]


COND_CHANNELS = ['support', 'force_x', 'force_y', 'vf_cap', 'dist_load', 'dist_support']


def condition_image(cond):
    """The condition drawn on the 80 x 80 element grid as 6 channels (COND_CHANNELS), for convolutional networks:
    1 on elements touching a clamped node; force_x and force_y on the elements of the load disc; vf_cap everywhere;
    distance from each element centre to the load point and to the nearest support element, in plate widths."""
    cond = np.asarray(cond, dtype=float)
    ex, ey = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5)          # element centres, indexed [y, x]
    disc = (ex - cond[0])**2 + (ey - cond[1])**2 <= (LOAD_R + 0.5)**2
    sup = support_elements(cond)
    return np.stack([sup, disc*cond[2], disc*cond[3], np.full((N, N), cond[4]),
                     np.hypot(ex - cond[0], ey - cond[1])/N, ndimage.distance_transform_edt(~sup)/N]).astype(np.float32)


def load_vectors(cond, design=None, extra_angles=()):
    """Force matrix (NDOF, 1 + len(extra_angles)): the unit load, plus the same load rotated by each extra angle
    (radians, counterclockwise as drawn), spread evenly over the disc nodes. With a design, only nodes touching
    material carry load; returns None if there are none."""
    nodes = load_nodes(cond)
    if design is not None:
        nodes = touching(nodes, design)
        if not nodes:
            return None
    fx, fy = cond[2], -cond[3]                       # y down in the condition, y dof up in the mesh
    F = np.zeros((NDOF, 1 + len(extra_angles)))
    n = node_id(*np.array(nodes).T)
    for k, a in enumerate((0.0,) + tuple(extra_angles)):
        F[2*n, k] += (np.cos(a)*fx - np.sin(a)*fy)/len(nodes)
        F[2*n + 1, k] += (np.sin(a)*fx + np.cos(a)*fy)/len(nodes)
    return F


def fixed_dofs(cond):
    n = node_id(*np.array(support_nodes(cond)).T)
    return np.unique(np.concatenate([2*n, 2*n + 1]))


# ---------------------------------------------------------------- FEA
def solve(density, cond, F, penal=1.0):
    """Displacements for each column of F, with element stiffness EMIN + density**penal."""
    xv = img2vec(density)
    sK = (KE.ravel()[None, :]*(EMIN + xv**penal*(1 - EMIN))[:, None]).ravel()
    K = sp.coo_matrix((sK, (_IK, _JK)), shape=(NDOF, NDOF)).tocsc()
    free = np.setdiff1d(np.arange(NDOF), fixed_dofs(cond))
    Kf = K[free][:, free]
    U = np.zeros_like(F)
    U[free] = spla.splu(((Kf + Kf.T)/2).tocsc(), permc_spec='MMD_AT_PLUS_A').solve(F[free])
    return U


def design_load(cond, design):
    """The unit load on the disc nodes that touch material of a binary design, or on the whole disc if none do."""
    F = load_vectors(cond, design)
    return load_vectors(cond) if F is None else F


def compliance(design, cond):
    """Compliance of the design thresholded at 0.5, under design_load (lower is stiffer). Empty space is modelled
    as material EMIN times as stiff, so the FEA always works; compliances above roughly 1e4 to 1e5 mean the load
    path is broken (material not connected to the load or to a support)."""
    b = np.nan_to_num(np.asarray(design, dtype=float)) > 0.5
    F = design_load(cond, b)
    return float(F[:, 0] @ solve(b.astype(float), cond, F)[:, 0])


def compliance_gradient(density, cond, penal=1.0):
    """Compliance of a density image in [0, 1] (as in compliance(), without thresholding; the load sits on the
    disc nodes touching density > 0.5) and its gradient dC/d(density), an (80, 80) image. The gradient is never
    positive: adding material never makes the plate softer."""
    x = np.clip(np.asarray(density, dtype=float), 0, 1)
    F = design_load(cond, x > 0.5)
    U = solve(x, cond, F, penal)[:, 0]
    ce = np.einsum('ij,jk,ik->i', U[EDOF], KE, U[EDOF])                 # strain energy of each element at unit stiffness
    return float(F[:, 0] @ U), vec2img(-penal*img2vec(x)**(penal - 1)*(1 - EMIN)*ce)


def compliance_gradients(densities, conds, penal=1.0):
    """compliance_gradient for a batch: compliances (n,) and gradients (n, 80, 80)."""
    out = [compliance_gradient(d, c, penal) for d, c in zip(densities, conds)]
    return np.array([C for C, _ in out]), np.stack([g for _, g in out])


def evaluate(designs, conds):
    """Compliance (FEA) of each design, and the volume fraction it uses relative to its cap: arrays of shape (n,)."""
    designs = np.asarray(designs, dtype=float) > 0.5
    conds = np.asarray(conds, dtype=float)
    return np.array([compliance(d, c) for d, c in zip(designs, conds)]), designs.mean(axis=(1, 2))/conds[:, 4]


# ---------------------------------------------------------------- reference SIMP (density filter, Heaviside continuation, OC)
_FILT = {}


def density_filter(rmin):
    if rmin not in _FILT:
        r = int(np.ceil(rmin)) - 1
        ex, ey = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        ex, ey = ex.ravel(), ey.ravel()
        rows, cols, vals = [], [], []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                w = rmin - np.hypot(dx, dy)
                ok = (ex + dx >= 0) & (ex + dx < N) & (ey + dy >= 0) & (ey + dy < N)
                if w > 0:
                    rows.append((ex*N + ey)[ok]); cols.append(((ex + dx)*N + ey + dy)[ok]); vals.append(np.full(ok.sum(), w))
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N*N, N*N))
        _FILT[rmin] = (H, np.asarray(H.sum(1)).ravel())
    return _FILT[rmin]


def simp(cond, rmin=2.0, maxit=100, penal=3.0, move=0.2, vf=None, x0=None):
    """Compliance-minimizing SIMP under the volume cap (or vf). Knows nothing about the hidden rules.
    x0: optional (80, 80) starting densities in [0, 1], e.g. a generated design; by default SIMP starts from a
    uniform plate at the volume cap. Returns an (80, 80) density image; threshold at 0.5 to submit."""
    vf = cond[4] if vf is None else vf
    H, Hs = density_filter(rmin)
    F = load_vectors(cond)[:, 0]
    free = np.setdiff1d(np.arange(NDOF), fixed_dofs(cond))
    x = np.full(N*N, vf) if x0 is None else np.clip(img2vec(np.asarray(x0, dtype=float)), 0, 1)
    beta, eta, itb = 1.0, 0.5, 0
    den = lambda b: np.tanh(b*eta) + np.tanh(b*(1 - eta))
    proj = lambda xt, b: (np.tanh(b*eta) + np.tanh(b*(xt - eta)))/den(b)
    for it in range(maxit):
        itb += 1
        xt = (H @ x)/Hs; xp = proj(xt, beta)
        sK = (KE.ravel()[None, :]*(EMIN + xp**penal*(1 - EMIN))[:, None]).ravel()
        K = sp.coo_matrix((sK, (_IK, _JK)), shape=(NDOF, NDOF)).tocsc()
        Kf = K[free][:, free]; U = np.zeros(NDOF); U[free] = spla.splu(((Kf + Kf.T)/2).tocsc()).solve(F[free])
        ce = np.einsum('ij,jk,ik->i', U[EDOF], KE, U[EDOF])
        dc = -penal*xp**(penal - 1)*(1 - EMIN)*ce
        dpx = beta*(1 - np.tanh(beta*(xt - eta))**2)/den(beta)
        dc = H @ (dc*dpx/Hs); dv = H @ (dpx/Hs)
        dc = np.minimum(dc, -1e-12); dv = np.maximum(dv, 1e-9)
        l1, l2 = np.log(1e-12), np.log(1e12)
        while l2 - l1 > 2e-3:
            lm = np.exp(0.5*(l1 + l2))
            xn = np.clip(x*np.sqrt(-dc/dv/lm), np.maximum(0, x - move), np.minimum(1, x + move))
            l1, l2 = (np.log(lm), l2) if proj((H @ xn)/Hs, beta).mean() > vf else (l1, np.log(lm))
        ch = np.abs(xn - x).max(); x = xn
        if beta < 16 and (itb >= 25 or ch < 0.01):
            beta *= 2; itb = 0
    return vec2img(proj((H @ x)/Hs, beta))


# ---------------------------------------------------------------- plotting
def plot_design(design, cond, ax=None, title=None):
    """Draw a design with its supports (blue), load disc and force direction (red)."""
    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax
    ax.imshow(np.asarray(design, dtype=float), cmap='gray_r', vmin=0, vmax=1, extent=(0, N, N, 0))
    for t in cond[5:8]:
        pts = np.array([perimeter_node(p) for p in range(int(round(t - SEG_LEN/2)), int(round(t + SEG_LEN/2)) + 1)])
        ax.plot(pts[:, 0], pts[:, 1], color='tab:blue', lw=4, solid_capstyle='butt', clip_on=False)
    ax.add_patch(plt.Circle((cond[0], cond[1]), LOAD_R, color='tab:red'))
    ax.arrow(cond[0], cond[1], 14*cond[2], 14*cond[3], color='tab:red', width=0.8, head_width=4, length_includes_head=True)
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_xticks([]); ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=8)
    return ax


def plot_designs(designs, conds, titles=None, ncols=6, size=2.3):
    """Draw designs with their conditions in a grid, ncols per row. Returns the axes."""
    import matplotlib.pyplot as plt
    nrows = -(-len(designs)//ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(size*ncols, (size + 0.2)*nrows), squeeze=False)
    for i, ax in enumerate(axs.flat):
        if i < len(designs):
            plot_design(designs[i], conds[i], ax, None if titles is None else titles[i])
        else:
            ax.axis('off')
    plt.tight_layout()
    return axs


def print_unet(model, in_shape=(7, N, N)):
    """Push a dummy input through a U-Net with .down and .up block lists (like Demo 8's) and draw its layout,
    with the output shape [channels x height x width] of every block."""
    import torch
    shapes, n = {}, len(model.down)
    blocks = [(f'down {k + 1}', b) for k, b in enumerate(model.down)] + [(f'up {n - 1 - k}', b) for k, b in enumerate(model.up)]
    hooks = [b.register_forward_hook(lambda m, inp, out, name=name: shapes.update({name: tuple(out.shape[1:]),
                                                                                    name + ' in': tuple(inp[0].shape[1:])}))
             for name, b in blocks]
    with torch.no_grad():
        device = next(model.parameters()).device
        x_out = model(torch.zeros(1, *in_shape, device=device), torch.zeros(1, dtype=torch.long, device=device))
    for h in hooks:
        h.remove()
    box = lambda s: '[%3d x %2d x %2d]' % s
    right = [56 - 8*k for k in range(n)]                          # column where each level's right-hand block starts
    print('Shapes are [channels x height x width]\n')
    print(f'input {box(tuple(in_shape))} -> {box(shapes["down 1 in"])}')
    print('│')
    for k in range(n - 1):
        left = f'down {k + 1}  {box(shapes[f"down {k + 1}"])} '
        line = left + ' skip '.center(right[k] - len(left) - 2, '─') + '▶ ' + f'up {k + 1}  {box(shapes[f"up {k + 1}"])}'
        print(line + (f' -> output {box(tuple(x_out.shape[1:]))}' if k == 0 else ''))
        print('│ avg pool'.ljust(right[k + 1]) + '▲ upsample')
    left = f'bottom  {box(shapes[f"down {n}"])} '
    print(left + '─'*(right[n - 1] - len(left)) + '┘')
    print(f'\nParameters: {sum(p.numel() for p in model.parameters()):,}')


# ---------------------------------------------------------------- I/O
N_CONCEPTS = 3          # designs per condition in a submission; the best of them counts


def load_designs(path, labels=False):
    """A dataset file: designs (n, 80, 80) float32 of 0/1, conditions (n, 8) and compliance (n,).
    With labels=True, also valid (n,) booleans (only in files that have them)."""
    with np.load(path, allow_pickle=False) as f:
        out = (np.unpackbits(f['designs'], axis=-1, count=N).astype(np.float32), f['conditions'], f['compliance'])
        return out + (f['valid'],) if labels else out


def load_round(path=None, folder='rounds'):
    """A round file: round id (str) and conditions (1000, 8).
    Without a path, loads the newest round (highest id) in folder, where the leaderboard's rNNNN.npz downloads go."""
    if path is None:
        names = sorted(f for f in os.listdir(folder) if re.fullmatch(r'r\d{4}\.npz', f))
        if not names:
            raise FileNotFoundError('no round files (rNNNN.npz) in %r' % folder)
        path = os.path.join(folder, names[-1])
    with np.load(path, allow_pickle=False) as f:
        round_id, conds = str(f['round_id']), f['conditions']
    print('loaded round %s from %s%s' % (round_id, path, ' (practice round, not accepted on the leaderboard)'
                                         if round_id == 'r0000' else ''))
    return round_id, conds


def save_submission(path, designs, round_id):
    """designs: (n_conditions, N_CONCEPTS, 80, 80), one set of concepts per condition of the round, in order.
    Thresholded at 0.5 and bit-packed (800 bytes per design)."""
    d = np.asarray(designs, dtype=float)
    if d.ndim != 4 or d.shape[1:] != (N_CONCEPTS, N, N):
        raise ValueError('designs must have shape (n_conditions, %d, %d, %d), got %s' % (N_CONCEPTS, N, N, d.shape))
    np.savez_compressed(path, round_id=str(round_id), designs=np.packbits(np.nan_to_num(d) > 0.5, axis=-1))


def load_submission(path):
    """Round id and designs (n_conditions, N_CONCEPTS, 80, 80) booleans."""
    with np.load(path, allow_pickle=False) as f:
        return str(f['round_id']), np.unpackbits(f['designs'], axis=-1, count=N).astype(bool)
