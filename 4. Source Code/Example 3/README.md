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

## Code

```python
import numpy as np
import matplotlib.pyplot as plt

# Objective functions
def f1(x, y):
    return x

def f2(x, y):
    return 1 - np.sqrt(x) + 0.3*np.sin(8*x) + y**2

# Random samples
samples = np.random.uniform([0,-1],[1,1],(2000,2))

x = samples[:,0]
y = samples[:,1]

vals_f1 = f1(x,y)
vals_f2 = f2(x,y)

# Pareto frontier (y=0)
x_pf = np.linspace(0,1,1000)
pareto_f1 = x_pf
pareto_f2 = 1 - np.sqrt(x_pf) + 0.3*np.sin(8*x_pf)

# Plot
plt.figure(figsize=(8,6))

plt.scatter(
    vals_f1,
    vals_f2,
    s=10,
    alpha=0.3,
    label="Sampled solutions"
)

plt.plot(
    pareto_f1,
    pareto_f2,
    'r',
    linewidth=3,
    label="Pareto frontier"
)

plt.xlabel("$f_1$")
plt.ylabel("$f_2$")
plt.title("Non-convex Pareto Frontier")
plt.legend()
plt.grid(True)
plt.show()
