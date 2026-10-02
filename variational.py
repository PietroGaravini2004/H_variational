import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Definition of the Gaussian exponents
alpha = np.array([
    13.00773,
    1.962079,
    0.444529,
    0.1219492
])

hartree_to_ev = 27.2114

# Number of basis functions
n = len(alpha)

# Initialize matrices
S = np.zeros((n, n))
T = np.zeros((n, n))
A = np.zeros((n, n))

# Fill the matrices
for p in range(n):
    for q in range(n):
        S[p, q] = (np.pi / (alpha[p] + alpha[q]))**(3/2)

        T[p, q] = (
            3 * alpha[p] * alpha[q] * np.pi**(3/2)
            / (alpha[p] + alpha[q])**(5/2)
        )

        A[p, q] = -2 * np.pi / (alpha[p] + alpha[q])

# Hamiltonian
H = T + A

# Radial grid for plotting
r = np.linspace(0, 5, 500)

# Exact hydrogen 1s wave function in atomic units
psi_exact = (1 / np.sqrt(np.pi)) * np.exp(-r)

# Plot
plt.figure(figsize=(8, 6))

for N in range(1, n + 1):
    # Truncated matrices
    H_N = H[:N, :N]
    S_N = S[:N, :N]

    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(H_N, S_N)

    # Ground-state energy
    E0 = eigenvalues[0]
    E0_eV = E0 * hartree_to_ev
    print(f"N = {N}, E0 = {E0_eV:.6f} eV")

    # Ground-state coefficients
    C = eigenvectors[:, 0]

    # Reconstruct the wave function
    psi_N = np.zeros_like(r)
    for p in range(N):
        psi_N += C[p] * np.exp(-alpha[p] * r**2)

    # Plot
    plt.plot(r, psi_N, label=f"N = {N}")

# Plot exact wave function
plt.plot(r, psi_exact, '--', label="Exact 1s")

plt.xlabel("r (a0)")
plt.ylabel(r"$\psi(r)$")
plt.title("Variational hydrogen wave function")
plt.legend()
plt.grid(True)
plt.show()