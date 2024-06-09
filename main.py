import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Define the function f(x)
def f(x):
    x_str = str(x)
    if '31415' in x_str:
        return min(3 / 4, max(1 / 4, (x ** 2 - 1) ** 2))
    else:
        return np.sin(x)


# Define the function F(x)
def F(x):
    if 0 <= x <= 2:
        return np.sqrt(x)
    elif 2 < x <= 5:
        return x * np.floor(x)


# Vectorize the functions for efficient numerical computations
f_vec = np.vectorize(f)
F_vec = np.vectorize(F)


# Define the sequence of simple functions f_n(x)
def f_n(x, n):
    # Partition the interval [0, 5] into n equal subintervals
    partitions = np.linspace(0, 5, n + 1)
    f_n_values = np.zeros_like(x)

    for i in range(n):
        mask = (x >= partitions[i]) & (x < partitions[i + 1])
        f_n_values[mask] = f(partitions[i])

    return f_n_values


# Define the range for x
x = np.linspace(0, 5, 1000)

# Plot f(x) and f_n(x) for several values of n
plt.figure(figsize=(10, 6))
plt.plot(x, f_vec(x), label='f(x)', color='blue')

for n in [5, 10, 20, 50, 100, 1000]:
    plt.plot(x, f_n(x, n), label=f'f_{n}(x)', linestyle='--')

plt.xlabel('x')
plt.ylabel('f(x) and f_n(x)')
plt.title('Plot of f(x) and f_n(x) for various n')
plt.legend()
plt.grid(True)
plt.show()


# Compute the Lebesgue integral using the simple functions f_n(x)
def lebesgue_integral(n):
    partitions = np.linspace(0, 5, n + 1)
    integral_sum = 0
    for i in range(n):
        x_i = partitions[i]
        delta_x = partitions[i + 1] - partitions[i]
        integral_sum += f_n(x_i, n) * delta_x
    return integral_sum


# Compute the Lebesgue-Stieltjes integral using the simple functions f_n(x)
def lebesgue_stieltjes_integral(n):
    partitions = np.linspace(0, 5, n + 1)
    integral_sum = 0
    for i in range(n):
        x_i = partitions[i]
        integral_sum += f_n(x_i, n) * (F_vec(partitions[i + 1]) - F_vec(partitions[i]))
    return integral_sum


# Calculate the Lebesgue and Lebesgue-Stieltjes integrals for several values of n
results = []
for n in [5, 10, 20, 50, 100, 1000]:
    lebesgue_int = lebesgue_integral(n)
    lebesgue_stieltjes_int = lebesgue_stieltjes_integral(n)
    results.append((n, lebesgue_int, lebesgue_stieltjes_int))
    print(f'n = {n}: Lebesgue integral = {lebesgue_int}, Lebesgue-Stieltjes integral = {lebesgue_stieltjes_int}')

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results, columns=['n', 'Lebesgue Integral', 'Lebesgue-Stieltjes Integral'])
