# Pareto Frontier - Simple Bi-objective Example

This project illustrates a **simple bi-objective optimization problem** with two quadratic distance functions, highlighting the concept of the **Pareto frontier**.

## Problem Definition

- Objective 1:  
  \( f_1(x) = x_1^2 + x_2^2 \) → distance from the origin  

- Objective 2:  
  \( f_2(x) = (x_1 - 2)^2 + (x_2 - 2)^2 \) → distance from the point (2,2)  

## What the Program Does

- **Generates random samples** in the decision space \([-5, 5]^2\)  
- **Evaluates** each sample with respect to both objectives  
- **Constructs** the theoretical Pareto frontier (line between (0,0) and (2,2))  
- **Plots** the sampled solutions together with the Pareto frontier  

## Expected Output

- A plot showing:  
  - Blue/gray points → sampled solutions  
  - Red line → theoretical Pareto frontier  

## Visualization

The plot demonstrates the trade-off between the two objectives:  
- Minimizing the distance to the origin  
- Minimizing the distance to the point (2,2)  

The **Pareto frontier** represents the set of solutions that cannot be improved in both objectives simultaneously.

---

## Code

```python

import numpy as np
import matplotlib.pyplot as plt

# Objective functions
def f1(x1, x2):
    return x1**2 + x2**2

def f2(x1, x2):
    return (x1 - 2)**2 + (x2 - 2)**2

# Random samples
samples = np.random.uniform(-5, 5, (2000, 2))
vals_f1 = f1(samples[:,0], samples[:,1])
vals_f2 = f2(samples[:,0], samples[:,1])

# Pareto frontier (theoretical line between (0,0) and (2,2))
t = np.linspace(0, 1, 200)
x1 = 2 * t
x2 = 2 * t
pareto_f1 = f1(x1, x2)
pareto_f2 = f2(x1, x2)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(vals_f1, vals_f2, s=10, alpha=0.3, label="Sampled solutions")
plt.plot(pareto_f1, pareto_f2, color='red', linewidth=2, label="Pareto frontier (theoretical)")
plt.xlabel("f1(x) = distance from origin")
plt.ylabel("f2(x) = distance from point (2,2)")
plt.title("Pareto Frontier - Simple Bi-objective Example")
plt.legend()
plt.grid(True)
plt.show()


