# Multiobjective Knapsack Problem (MOKP) - Pareto Front Example

This project demonstrates a **bi-objective unconstrained knapsack problem** and computes the **Pareto front**.

## Problem Definition

Bi-objective unconstrained knapsack:

- Maximize:  
  \( z_1(x) = \sum c_j \cdot x_j \) → total profit  
  \( z_2(x) = W - \sum w_j \cdot x_j \) → remaining capacity  

- Subject to:  
  \( x_j \in \{0,1\}, \; j = 1..n \)

The program:
- Enumerates all \(2^n\) combinations
- Computes \((z_1, z_2)\) for each solution
- Identifies non-dominated (Pareto-optimal) solutions
- Prints them in tabular form
- Plots the full solution set with the Pareto frontier

---

## Code

```python
import itertools
import matplotlib.pyplot as plt

# Problem data
W = 100
weights = [25, 15, 14, 25, 15]   # w_j
profits = [80, 120, 80, 15, 30]  # c_j
n = len(weights)

def evaluate(x):
    z1 = sum(c * xi for c, xi in zip(profits, x))
    z2 = W - sum(w * xi for w, xi in zip(weights, x))
    return z1, z2

def dominates(sol_a, sol_b):
    z1a, z2a = sol_a
    z1b, z2b = sol_b
    better_or_equal = z1a >= z1b and z2a >= z2b
    strictly_better = z1a > z1b or z2a > z2b
    return better_or_equal and strictly_better

def pareto_front(solutions):
    non_dominated = []
    for i, sol_i in enumerate(solutions):
        dominated = False
        for j, sol_j in enumerate(solutions):
            if i == j:
                continue
            if dominates(sol_j, sol_i):
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)
    return non_dominated

def main():
    all_x = list(itertools.product([0, 1], repeat=n))
    all_solutions = [evaluate(x) for x in all_x]

    print(f"Total number of solutions evaluated: {len(all_solutions)}\n")

    nd_idx = pareto_front(all_solutions)
    nd_idx.sort(key=lambda i: all_solutions[i][0])

    print("Non-dominated solutions (Pareto front):")
    print(f"{'x (selection)':<20}{'z1 (profit)':<15}{'z2 (remaining cap.)':<20}")
    for i in nd_idx:
        z1, z2 = all_solutions[i]
        x_str = "".join(map(str, all_x[i]))
        print(f"{x_str:<20}{z1:<15}{z2:<20}")

    all_z1 = [s[0] for s in all_solutions]
    all_z2 = [s[1] for s in all_solutions]
    nd_z1 = [all_solutions[i][0] for i in nd_idx]
    nd_z2 = [all_solutions[i][1] for i in nd_idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(all_z1, all_z2, c="steelblue", label="All solutions", zorder=2)
    plt.scatter(nd_z1, nd_z2, c="red", label="Non-dominated (Pareto front)", zorder=3)
    plt.plot(nd_z1, nd_z2, "r-", linewidth=1.5, zorder=1)

    plt.xlabel("$z_1(X)$ - Total profit")
    plt.ylabel("$z_2(X)$ - Remaining capacity")
    plt.title("Pareto Front - Multiobjective Knapsack Problem")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("mokp_pareto_front.png", dpi=200)
    print("\nPlot saved as 'mokp_pareto_front.png'")
    plt.show()

if __name__ == "__main__":
    main()
