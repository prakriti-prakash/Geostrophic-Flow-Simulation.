"""
geoflow/shallow_water.py
========================
A small rotating shallow-water simulator.

Why this exists
---------------
Your original notebook turns a sea-surface slope into a current using
geostrophic balance. That balance is what the ocean looks like AFTER the water
has settled. This simulator shows how it gets there:

   1. Water is piled up (a hill of water). Gravity pulls it DOWN the slope.
   2. As soon as it moves, the Earth's rotation (Coriolis force) pushes it
      sideways (to the right in the Northern Hemisphere).
   3. Some energy leaves as gravity waves (the ripples spreading outwards).
   4. What is left is a steady flow that runs ALONG the height contours - the
      geostrophic current from your notebook.

Equations (linear, f-plane, flat bottom):

    du/dt =  f v - g d(eta)/dx
    dv/dt = -f u - g d(eta)/dy
    d(eta)/dt = -H ( du/dx + dv/dy )

eta = surface height anomaly, (u, v) = velocity, H = layer depth,
f = Coriolis parameter, g = gravity (use a reduced g for a thin ocean layer).

Numerics: Arakawa C-grid, forward-backward time stepping, periodic edges with
a "sponge" that absorbs waves near the boundary so they do not wrap around.
"""

import numpy as np


class RotatingShallowWater:
    def __init__(self, nx, ny, dx, dt, f=1.0e-4, g=0.03, H=500.0,
                 sponge_cells=0, sponge_hours=3.0):
        self.nx, self.ny, self.dx, self.dt = nx, ny, dx, dt
        self.f, self.g, self.H = f, g, H
        self.c = np.sqrt(g * H)                 # gravity-wave speed (m/s)
        self.rossby_radius = self.c / f         # natural length scale (m)
        if self.c * dt / dx > 0.5:
            raise ValueError("time step too large for stability: "
                             f"c*dt/dx = {self.c * dt / dx:.2f} (keep below 0.5)")

        self.eta = np.zeros((ny, nx))           # at cell centres
        self.u = np.zeros((ny, nx))             # on east faces
        self.v = np.zeros((ny, nx))             # on north faces
        self.eta_ref = np.zeros((ny, nx))       # sponge relaxes towards this
        self.time = 0.0                         # seconds

        # sponge: damping that grows towards the domain edge
        self.damping = np.zeros((ny, nx))
        if sponge_cells > 0:
            j, i = np.mgrid[0:ny, 0:nx]
            edge = np.minimum.reduce([i, nx - 1 - i, j, ny - 1 - j]).astype(float)
            ramp = np.clip((sponge_cells - edge) / sponge_cells, 0, 1) ** 2
            self.damping = ramp / (sponge_hours * 3600.0)

    # ---------------------------------------------------------------------
    def set_initial_height(self, eta0):
        self.eta[:] = eta0
        self.eta_ref[:] = eta0                  # far-field level to hold at the edges
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.time = 0.0

    def step(self):
        dt, dx, f, g, H = self.dt, self.dx, self.f, self.g, self.H
        u, v, eta = self.u, self.v, self.eta

        # 1) height changes where the flow converges / diverges
        div = (u - np.roll(u, 1, axis=1) + v - np.roll(v, 1, axis=0)) / dx
        eta -= dt * H * div
        eta -= dt * self.damping * (eta - self.eta_ref)

        # 2) east-west flow: Coriolis from v, push from the height slope
        v_at_u = 0.25 * (v + np.roll(v, -1, axis=1) + np.roll(v, 1, axis=0)
                         + np.roll(np.roll(v, 1, axis=0), -1, axis=1))
        u += dt * (f * v_at_u - g * (np.roll(eta, -1, axis=1) - eta) / dx)

        # 3) north-south flow: Coriolis from the NEW u
        u_at_v = 0.25 * (u + np.roll(u, 1, axis=1) + np.roll(u, -1, axis=0)
                         + np.roll(np.roll(u, -1, axis=0), 1, axis=1))
        v += dt * (-f * u_at_v - g * (np.roll(eta, -1, axis=0) - eta) / dx)

        # 4) sponge on the velocities
        damp = 1.0 - dt * self.damping
        u *= damp
        v *= damp
        self.time += dt

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    # ---------------------------------------------------------------------
    def velocity_at_centres(self):
        """u, v averaged onto the cell centres (for plotting / tracers)."""
        return (0.5 * (self.u + np.roll(self.u, 1, axis=1)),
                0.5 * (self.v + np.roll(self.v, 1, axis=0)))

    def energy(self):
        """Total energy (J per metre of depth-equivalent); conserved without sponge."""
        return 0.5 * self.dx ** 2 * float(np.sum(self.H * (self.u ** 2 + self.v ** 2)
                                                 + self.g * self.eta ** 2))


# ---------------------------------------------------------------------------
# Starting shapes (x, y in metres from the domain centre)
# ---------------------------------------------------------------------------
def initial_height(kind, nx, ny, dx, amplitude=60.0, size=70e3):
    """
    kind = 'mound'  : one hill of water
           'dipole' : a hill next to a hollow (water flows between them)
           'ridge'  : a long wall of water
    """
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dx
    X, Y = np.meshgrid(x, y)
    if kind == "mound":
        return amplitude * np.exp(-(X ** 2 + Y ** 2) / (2 * size ** 2))
    if kind == "dipole":
        shift = 1.35 * size
        hill = np.exp(-((X + shift) ** 2 + Y ** 2) / (2 * size ** 2))
        hole = np.exp(-((X - shift) ** 2 + Y ** 2) / (2 * size ** 2))
        return amplitude * (hill - hole)
    if kind == "ridge":
        return amplitude * np.exp(-X ** 2 / (2 * size ** 2))
    raise ValueError(f"unknown shape '{kind}'")


# ---------------------------------------------------------------------------
# Check against the textbook solution (Rossby adjustment, Gill 1982)
# ---------------------------------------------------------------------------
def rossby_adjustment_check(eta0=1.0, f=1.0e-4, g=0.03, H=500.0, dx=2500.0,
                            width_cells=1.0, length=3000e3, days=2.0):
    """
    Release a 'dam' of water: height +eta0 on the left, -eta0 on the right.
    Theory (Gill, 1982) says that after adjustment the height step is replaced by

        eta(x) = -eta0 * sign(x) * (1 - exp(-|x| / Rd))        Rd = sqrt(gH)/f

    with a jet along the front of peak speed  |v| = eta0 * sqrt(g / H).
    This runs the simulator and compares both predictions.

    The model's peak speed is a little low because the starting step is spread
    over `width_cells` grid cells instead of being perfectly sharp. The error
    shrinks steadily as dx and width_cells are reduced.
    """
    nx = int(length / dx)
    c = np.sqrt(g * H)
    sim = RotatingShallowWater(nx=nx, ny=4, dx=dx, dt=0.1 * dx / c, f=f, g=g, H=H)
    L = nx * dx
    x = (np.arange(nx) - nx / 4.0) * dx            # runs from -L/4 to 3L/4
    width = width_cells * dx
    # The domain is periodic, so a single step would need a second, opposite
    # step to wrap around. Put the one we measure at x = 0 and the other at
    # x = L/2, far enough away that their waves do not meet during the run.
    eta_line = eta0 * np.where(x < L / 4, -np.tanh(x / width), np.tanh((x - L / 2) / width))
    sim.set_initial_height(np.tile(eta_line, (4, 1)))

    # average over the last inertial period to remove the leftover oscillation
    n_total = int(days * 86400 / sim.dt)
    n_avg = int(2 * np.pi / f / sim.dt)
    sim.run(n_total - n_avg)
    eta_sum = np.zeros(nx)
    v_sum = np.zeros(nx)
    for _ in range(n_avg):
        sim.step()
        eta_sum += sim.eta[1]
        v_sum += sim.velocity_at_centres()[1][1]
    eta_num, v_num = eta_sum / n_avg, v_sum / n_avg

    Rd = sim.rossby_radius
    near = np.abs(x) < 4 * Rd
    eta_theory = -eta0 * np.sign(x) * (1 - np.exp(-np.abs(x) / Rd))
    v_peak_theory = eta0 * np.sqrt(g / H)
    v_peak_model = float(np.max(np.abs(v_num[near])))
    return {
        "rossby_radius_km": Rd / 1e3,
        "v_peak_theory": v_peak_theory,
        "v_peak_model": v_peak_model,
        "v_peak_error": v_peak_model / v_peak_theory - 1.0,
        "eta_rel_error": float(np.sqrt(np.mean((eta_num[near] - eta_theory[near]) ** 2)
                                       / np.mean(eta_theory[near] ** 2))),
        "x_km": x[near] / 1e3, "eta_model": eta_num[near], "eta_theory": eta_theory[near],
        "v_model": v_num[near],
    }
