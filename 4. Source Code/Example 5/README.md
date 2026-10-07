# 🚚 Bi-Objective Traveling Salesperson Problem (Bi-TSP)

An exact implementation of the **Augmented $\epsilon$-Constraint Method (AUGMECON2 style)** to solve a Bi-Objective Traveling Salesperson Problem (Bi-TSP, $n = 10$) using Python's free open-source MILP solver **HiGHS** (`scipy.milp`).

---

## 📌 Problem Overview

The Bi-Objective Traveling Salesperson Problem seeks to minimize two conflicting objective functions over a set of $n$ cities:

$$\min \quad (f_1(x), f_2(x))$$

Where:
* **$f_1(x)$**: Total travel distance (primary objective).
* **$f_2(x)$**: Total travel cost (secondary objective).
* **Constraints**: $x$ must form a valid Hamiltonian tour using the Miller-Tucker-Zemlin (MTZ) subtour elimination formulation.

---

## 🛠️ Mathematical Method: Augmented $\epsilon$-Constraint

Instead of scalarizing objectives into a single weighted sum (which misses non-convex Pareto regions), we solve a sequence of single-objective subproblems:

$$\begin{aligned} \min \quad & f_1(x) - \rho \cdot \frac{s}{r_2} \\ \text{s.t.} \quad & f_2(x) + s = \epsilon \\ & s \ge 0 \\ & x \in X \quad \text{(valid TSP tours)} \end{aligned}$$

### Key Features of AUGMECON2:
1. **Augmentation Term ($\rho$):** Guarantees weakly efficient solutions are avoided by forcing positive slack $s$.
2. **Exact Grid Jumping:** Since costs are integers, after finding an efficient solution $x^*$ with $f_2^* = f_2(x^*)$, the next bound is set to $\epsilon = f_2^* - 1$. This automatically skips redundant subproblems.

---

## 🚀 Getting Started

### Prerequisites

Install the required Python libraries:

```bash
pip install numpy scipy matplotlib

# Code

```python
"""
Bi-objective TSP (n = 10) solved with the exact epsilon-constraint method
(augmented / AUGMECON2-style) using the free HiGHS MILP solver (scipy.milp).

    pip install numpy scipy matplotlib

Problem
-------
    min  ( f1(x), f2(x) )      f1 = total distance, f2 = total cost
    s.t. x is a Hamiltonian tour (MTZ subtour-elimination formulation)

Epsilon-constraint (minimisation form, f1 = primary objective):
    min   f1(x) - rho * s / r2
    s.t.  f2(x) + s = eps,   s >= 0,   x in X
Costs are integers, so the eps grid is exact: after a solve with
f2* = f2(x*), the next bound is eps = f2* - 1 (AUGMECON2 "skip": every eps in
(f2*, eps_old] would return the same point). Stop when the model is infeasible.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds

N = 10
SEED = 7
RHO = 1e-4          # augmentation coefficient (slide: rho in [1e-6, 1e-3])


# ------------------------------------------------------------------
# 1. Instance: two integer, symmetric cost matrices
# ------------------------------------------------------------------
def build_instance(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(0, 100, (n, 2))     # coordinates -> distance (f1)
    p2 = rng.uniform(0, 100, (n, 2))     # other coordinates -> cost (f2)
    d1 = np.rint(np.linalg.norm(p1[:, None] - p1[None], axis=2)).astype(int)
    d2 = np.rint(np.linalg.norm(p2[:, None] - p2[None], axis=2)).astype(int)
    return p1, p2, d1, d2


# ------------------------------------------------------------------
# 2. Model (MTZ formulation) in matrix form
#    variables: x_ij (n(n-1) binaries) | u_i, i=1..n-1 | s (slack)
# ------------------------------------------------------------------
class BiTSP:
    def __init__(self, d1, d2):
        self.n = n = len(d1)
        self.arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
        self.na = len(self.arcs)
        self.nv = self.na + (n - 1) + 1
        self.i_s = self.nv - 1
        self.c1 = np.zeros(self.nv)
        self.c2 = np.zeros(self.nv)
        for k, (i, j) in enumerate(self.arcs):
            self.c1[k], self.c2[k] = d1[i][j], d2[i][j]

        rows, lo, hi = [], [], []
        for i in range(n):                           # leave / enter once
            r = np.zeros(self.nv); r[[k for k, (a, b) in enumerate(self.arcs) if a == i]] = 1
            rows.append(r); lo.append(1); hi.append(1)
            r = np.zeros(self.nv); r[[k for k, (a, b) in enumerate(self.arcs) if b == i]] = 1
            rows.append(r); lo.append(1); hi.append(1)
        for k, (i, j) in enumerate(self.arcs):       # MTZ: u_i - u_j + (n-1)x_ij <= n-2
            if i != 0 and j != 0:
                r = np.zeros(self.nv)
                r[self.na + i - 1], r[self.na + j - 1], r[k] = 1, -1, n - 1
                rows.append(r); lo.append(-np.inf); hi.append(n - 2)
        self.A, self.lo, self.hi = np.array(rows), np.array(lo), np.array(hi)

        lb = np.zeros(self.nv); ub = np.ones(self.nv)
        lb[self.na:self.na + n - 1] = 1; ub[self.na:self.na + n - 1] = n - 1
        ub[self.i_s] = np.inf
        self.bounds = Bounds(lb, ub)
        integ = np.zeros(self.nv); integ[:self.na] = 1     # only x is binary
        self.integrality = integ

    def solve(self, c, extra=()):
        """extra: list of (row, lo, hi). Returns x or None if infeasible."""
        A, lo, hi = self.A, self.lo, self.hi
        if extra:
            A = np.vstack([A] + [e[0] for e in extra])
            lo = np.concatenate([lo, [e[1] for e in extra]])
            hi = np.concatenate([hi, [e[2] for e in extra]])
        res = milp(c, constraints=LinearConstraint(A, lo, hi),
                   bounds=self.bounds, integrality=self.integrality,
                   options={"mip_rel_gap": 0})
        return (res.x if res.status == 0 else None)

    def tour(self, x):
        succ = {i: j for k, (i, j) in enumerate(self.arcs) if x[k] > 0.5}
        t, cur = [0], succ[0]
        while cur != 0:
            t.append(cur); cur = succ[cur]
        return t

    def values(self, x):
        return round(self.c1 @ x), round(self.c2 @ x)


# ------------------------------------------------------------------
# 3. Lexicographic optimisation -> ideal and nadir (on the efficient set)
# ------------------------------------------------------------------
def lexmin(m, first, second):
    cf, cs = (m.c1, m.c2) if first == 1 else (m.c2, m.c1)
    x = m.solve(cf)
    best = round(cf @ x)
    x = m.solve(cs, extra=[(cf, best, best)])
    return x


# ------------------------------------------------------------------
# 4. Augmented epsilon-constraint loop
# ------------------------------------------------------------------
def epsilon_constraint(d1, d2):
    m = BiTSP(d1, d2)
    xa = lexmin(m, 1, 2)                       # best f1, ties broken by f2
    xb = lexmin(m, 2, 1)                       # best f2, ties broken by f1
    (z1, f2_max), (f1_max, z2) = m.values(xa), m.values(xb)
    r2 = max(f2_max - z2, 1)
    print(f"Ideal point : f1* = {z1}, f2* = {z2}")
    print(f"Nadir point : f1  = {f1_max}, f2  = {f2_max}\n")

    front = [(z1, f2_max, m.tour(xa))]         # extreme point (min f1)
    eps, n_solves = f2_max - 1, 0
    while eps >= z2:
        row = m.c2.copy(); row[m.i_s] = 1      # f2(x) + s = eps
        obj = m.c1.copy(); obj[m.i_s] = -RHO / r2
        x = m.solve(obj, extra=[(row, eps, eps)])
        n_solves += 1
        if x is None:
            break
        v1, v2 = m.values(x)
        front.append((v1, v2, m.tour(x)))
        print(f"  eps = {eps:4d} -> f1 = {v1:4d}, f2 = {v2:4d}")
        eps = v2 - 1                           # AUGMECON2 skip

    pts = sorted({(a, b): t for a, b, t in front}.items())   # dominance filter
    nd, best_f2 = [], float("inf")
    for (a, b), t in pts:
        if b < best_f2:
            nd.append((a, b, t)); best_f2 = b
    return nd, n_solves


# ------------------------------------------------------------------
# 5. Run, report and plot
# ------------------------------------------------------------------
if __name__ == "__main__":
    p1, p2, d1, d2 = build_instance()
    print(f"Bi-objective TSP, n = {N}, seed = {SEED}\n")
    t0 = time.time()
    front, n_solves = epsilon_constraint(d1, d2)
    elapsed = time.time() - t0

    print(f"\nPareto front: {len(front)} non-dominated solutions "
          f"({n_solves} epsilon-subproblems, {elapsed:.1f} s)\n")
    print(f"{'#':>3} {'f1 (dist)':>10} {'f2 (cost)':>10}  tour")
    for k, (a, b, t) in enumerate(front, 1):
        print(f"{k:>3} {a:>10} {b:>10}  {' -> '.join(map(str, t + [0]))}")

    # ---------------- figure ----------------
    F = np.array([(a, b) for a, b, _ in front])
    best_f1, best_f2 = front[0], front[-1]          # extreme solutions

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1])

    # Pareto front (left, spans both rows)
    axp = fig.add_subplot(gs[:, 0])
    axp.step(F[:, 0], F[:, 1], where="post", color="gray", lw=0.8, ls="--")
    axp.scatter(F[:, 0], F[:, 1], c="lightgray", s=50, edgecolor="k", zorder=3)
    axp.scatter(*best_f1[:2], c="tab:blue", s=130, edgecolor="k", zorder=4,
                label=f"Route A: min f1 = ({best_f1[0]}, {best_f1[1]})")
    axp.scatter(*best_f2[:2], c="tab:red", s=130, marker="s", edgecolor="k",
                zorder=4, label=f"Route B: min f2 = ({best_f2[0]}, {best_f2[1]})")
    axp.set_xlabel("f1 (total distance)"); axp.set_ylabel("f2 (total cost)")
    axp.set_title("Pareto front - epsilon-constraint (exact)")
    axp.grid(alpha=0.3); axp.legend(fontsize=9)

    def draw_route(ax, pts, tour, color, title):
        t = tour + [0]
        ax.plot(pts[t, 0], pts[t, 1], "-o", color=color, lw=1.8, ms=6)
        ax.scatter(*pts[0], s=140, facecolor="none", edgecolor="k", lw=2)  # depot 0
        for i, (px, py) in enumerate(pts):
            ax.annotate(str(i), (px, py), xytext=(5, 5), textcoords="offset points")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)

    # rows: routes A and B | columns: map of f1 (p1) and map of f2 (p2)
    draw_route(fig.add_subplot(gs[0, 1]), p1, best_f1[2], "tab:blue",
               f"Route A (min f1) on f1-map\nf1={best_f1[0]}")
    draw_route(fig.add_subplot(gs[0, 2]), p2, best_f1[2], "tab:blue",
               f"Route A (min f1) on f2-map\nf2={best_f1[1]}")
    draw_route(fig.add_subplot(gs[1, 1]), p1, best_f2[2], "tab:red",
               f"Route B (min f2) on f1-map\nf1={best_f2[0]}")
    draw_route(fig.add_subplot(gs[1, 2]), p2, best_f2[2], "tab:red",
               f"Route B (min f2) on f2-map\nf2={best_f2[1]}")

    plt.tight_layout()
    plt.savefig("bitsp_pareto.png", dpi=150)
    plt.show()
```
