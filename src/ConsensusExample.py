import numpy as np
from scipy.optimize import minimize_scalar

def admm_consensus(cost_functions, demand, rho=1.0, max_iters=50, tol=1.0e-4):
    N = len(cost_functions)

    x = np.zeros(N); y = np.zeros(N); z = np.zeros(N)

    for iter in range(max_iters):

        for i in range(N):

            def xi_objective(xi):
                return (
                    cost_functions[i](xi)
                    + y[i] * (xi - z[i])
                    + rho / 2 * (xi - z[i]) ** 2
                )

            x[i] = minimize_scalar(xi_objective).x

        temp_vec = y / rho + x
        z = temp_vec - (temp_vec.sum() - demand) / N
        
        y += rho*(x-z)

        print(f"Iteration = {iter+1}")
        
        res_norm = np.linalg.norm(x-z)
        if res_norm < tol:
            break
        
    return x, z


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
    x, z = admm_consensus(cost_functions, demand)

    # Print results
    print("Power generation by each provider:", x)
    print("Consensus power allocation:", z)
    print("Total power generated:", np.sum(x))
