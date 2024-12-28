import numpy as np
from scipy.optimize import minimize_scalar

def admm_energy_system(cost_functions, demand, rho=1.0, max_iters=50, tol=1e-4):
    """
    ADMM for distributed energy system optimization.

    Args:
        cost_functions: List of cost function callables f_i(x) for each provider.
        demand: Total power demand (D).
        rho: Augmented Lagrangian parameter.
        max_iters: Maximum number of iterations.
        tol: Convergence tolerance.

    Returns:
        x: Power generation for each provider.
        z: Consensus power allocation.
    """
    N = len(cost_functions)  # Number of providers

    # Initialize variables
    x = np.zeros(N)  # Local variables
    z = np.zeros(N)  # Global consensus variables
    u = np.zeros(N)  # Dual variables

    for k in range(max_iters):
        # Update x_i: Minimize local cost function + penalty term
        for i in range(N):
            # Local quadratic subproblem: minimize f_i(x_i) + (rho / 2) * (x_i - z[i] + u[i])^2
            def local_objective(x_i):
                return cost_functions[i](x_i) + (rho / 2) * (x_i - z[i] + u[i])**2
            x[i] = minimize_scalar(local_objective).x  # Use a scalar minimizer

        # Update z: Enforce consensus and global constraint
        z_old = z.copy()
        z = (x + u) - (np.sum(x + u) - demand) / N  # Ensure sum(z) = demand

        # Update u: Dual variable update
        u += x - z

        # Check convergence
        r_norm = np.linalg.norm(x - z)
        s_norm = np.linalg.norm(rho * (z - z_old))
        if r_norm < tol and s_norm < tol:
            break

    return x, z

# Example usage
if __name__ == "__main__":
    # Define cost functions for each provider
    cost_functions = [
        lambda x: 0.5 * x**2,  # Provider 1: Quadratic cost
        lambda x: 0.8 * x**2,  # Provider 2: Higher cost
        lambda x: 0.3 * x**2 + 2 * abs(x),  # Provider 3: Includes fixed cost
    ]

    # Total demand
    demand = 100

    # Solve using ADMM
    x, z = admm_energy_system(cost_functions, demand)

    # Print results
    print("Power generation by each provider:", x)
    print("Consensus power allocation:", z)
    print("Total power generated:", np.sum(x))
