import numpy as np


def sparsify_update_vector(algo_sandbox, step_size, grads, adapt_indices, power=10, min_update=0.1):
    """
    Args:
        algo_sandbox: AlgoSandbox object that contains all optimization parameters
        step_size: a float or np.array that represents the step size of each parameter in the sgd.
        grads: gradients calculated in the sgd iteration.
        adapt_indices: the dipole indices to adapt in this iteration.
        power: the power by which the error is raised in the loss function
        min_update: the lower bound by which the coeffs will be updated.

    Returns: new step sizes for each parameter such that the parameters that influence the most have larger step sizes.

    """

    # calculate the influence of each coeff on the current sample of the simulation
    influence = np.abs(grads * algo_sandbox.coeffs[adapt_indices]) ** power

    # normalize such that the parameters that influences the most has step_size=step_size,
    # and the parameters that does influence at all has min_update
    influence += np.max([influence.max() * min_update, 1e-20])
    influence /= max(influence.max(), 1e-15)

    return influence * step_size
