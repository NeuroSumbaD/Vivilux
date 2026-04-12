'''Submodule defining calibration routings for hardware meshes.
'''

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from vivilux.hardware.arbitrary_mzi import HardMesh

type StepGenerator = Callable[[int], np.ndarray] # Type alias for step vector generator

def gen_from_uniform(shape: tuple[int, ...]) -> StepGenerator:
    '''Generates step vectors from a uniform distribution between -0.5 and 0.5.
    '''
    def _generator(num_vectors = 1) -> np.ndarray:
        return (np.random.rand(num_vectors, *shape)-0.5)
    return _generator

def gen_from_sparse_permutation(shape: tuple[int, ...], numHot: int) -> StepGenerator:
    '''Generates step vectors with a fixed number of hot (1.0) entries at
        random positions.
        
        NOTE: Because numHot is an additional parameter, this generator must be
        used with `Partial` from the `functools` module to fix the numHot value.
        e.g. `step_generator=partial(gen_from_sparse_permutation, numHot=3)`
    '''
    def _generator(num_vectors=1) -> np.ndarray:
        arr = np.zeros((num_vectors, *shape))
        for i in range(num_vectors):
            hot_indices = np.random.choice(np.prod(shape), size=numHot, replace=False)
            np.put(arr[i], hot_indices, 1.0)
        return arr
    return _generator

def gen_from_one_hot(shape: tuple[int, ...]) -> StepGenerator:
    '''Generates one-hot step vectors.
    
        NOTE: Using sparse permutation with numHot=1 would achieve the same
        effect but may generate duplicate vectors.
    '''
    def _generator(num_vectors=1) -> np.ndarray:
        arr = np.eye(np.prod(shape))[np.random.permutation(np.prod(shape))]
        arr = arr[:num_vectors].reshape((num_vectors, *shape))
        return arr
    return _generator

def least_squares_solver(X: np.ndarray, V: np.ndarray,
                         deltaFlat: np.ndarray,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # minimize least squares difference to deltas
    a = None
    for iteration in range(X.shape[1]):
        xtx = X.T @ X
        rank = np.linalg.matrix_rank(xtx)
        if rank == len(xtx): # matrix will have an inverse
            a = np.linalg.inv(xtx) @ X.T @ deltaFlat
            break
        else: # direction vectors cary redundant information use one less
            X = X[:,:-1]
            V = V[:,:-1]
            continue
    if a is None:
        raise RuntimeError("ERROR: Could not compute least squares solution")
    return a, X, V

def least_squares_solver_v2(X: np.ndarray, V: np.ndarray,
                            deltaFlat: np.ndarray,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xtx = X.T @ X
    a = np.linalg.pinv(xtx) @ X.T @ deltaFlat
    return a, X, V

def get_directions(moment: np.ndarray,
                   num_directions: int = 6,
                   bias_scale: float=0.1,
                   ) -> np.ndarray:
    '''Rather than using random directions, we use a set of random directions
        that are biased by the current momentum term from Adam to help guide
        the search space to a consistent direction in parameter space.
    '''
    directions = np.random.uniform(low = -1, # Normalization takes care of scaling
                                   high = 1, # Uniform gives more orthogonal sample directions
                                   size=(len(moment), num_directions)) # shape (num_params, num_directions)
    directions += bias_scale * moment[:, np.newaxis]  # shape (num_params, num_directions)
    # Normalize to unit vectors (along each column) for consistent step sizes
    directions /= np.linalg.norm(directions, axis=0)[np.newaxis, :]  # shape (num_params, num_directions)
    return directions

def adam_lamm(init_delta: np.ndarray,
              hardware_mesh: HardMesh, 
              num_directions: int,
              direction_magnitude: float,
              learning_rate: float = 0.1,
              lr_decay: float = 0.99,
              beta1: float = 0.9,
              beta2: float = 0.99,
              epsilon: float = 1e-8,
              num_iterations: int = 200,
              bias_scale: float = 0.5,
              convergence_threshold: float = 1e-3,
              ):
    # Initialize Adam variables
    m = np.zeros(hardware_mesh.num_params)  # First moment
    v = np.zeros(hardware_mesh.num_params)  # Second moment

    # Initialize history with NaN values to indicate uninitialized entries
    history = np.full(num_iterations + 1, np.nan)
    current_delta = init_delta.copy()
    history[0] = np.linalg.norm(current_delta)  # Record initial delta magnitude

    params = hardware_mesh.get_params()
    target_matrix = hardware_mesh.get() + init_delta

    for iteration in range(num_iterations):
        optimal_step = np.zeros(hardware_mesh.num_params)
        
        # LSO of several directional derivatives to find optimal step
        directions = get_directions(m,
                                    num_directions=num_directions,
                                    bias_scale=bias_scale) # shape (num_params, num_directions)
        directions *= direction_magnitude  # Scale to desired step size
        derivatives = np.zeros((current_delta.size, num_directions))  # shape (output_size, num_directions)
        for dir_index, direction in enumerate(directions.T):
            derivatives[:, dir_index] = hardware_mesh.measure_directional_derivative(params, direction).flatten()
        
        # Solve least squares for optimal step
        a = np.linalg.pinv(derivatives.T @ derivatives) @ derivatives.T  @ current_delta.flatten() # shape (num_directions,)
        optimal_step = (directions @ a).flatten()  # shape (num_params,)

        # Adam update
        t = iteration + 1
        m = beta1 * m + (1 - beta1) * optimal_step
        v = beta2 * v + (1 - beta2) * optimal_step**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        lr_hat = learning_rate / (np.sqrt(v_hat) + epsilon)

        # Update parameter
        params += lr_hat * m_hat

        # Decay learning rate
        learning_rate *= lr_decay

        # Reflect momentum for parameters that hit boundaries
        hit_lower = (params < hardware_mesh.param_limits[0])
        hit_upper = (params > hardware_mesh.param_limits[1])
        hit_boundary = hit_lower | hit_upper
        
        if np.any(hit_boundary):
            m[hit_boundary] *= -0.8  # Reflect first moment (momentum)
            v[hit_boundary] = 0.0  # Reset second moment (adaptive learning rate)
    

        params = hardware_mesh.clip_params(params)  # Ensure parameters are within limits after update

        if np.any(np.isnan(params)):
            raise ValueError("NaN encountered in parameters during optimization.")

        hardware_mesh.set_params(params)

        # Measure new matrix and error
        new_meas = hardware_mesh.measure_matrix()
        new_delta = target_matrix - new_meas
        new_delta_mag = np.linalg.norm(new_delta)
        history[iteration + 1] = new_delta_mag
        current_delta = new_delta.copy()

        if new_delta_mag < convergence_threshold:  # Early stopping threshold
            break

    history = history[~np.isnan(history)]  # Remove uninitialized entries
    return history, params

def central_difference_descent(init_delta: np.ndarray,
                               hardware_mesh: HardMesh, 
                               learning_rate: float = 0.1,
                               delta_voltage: float = 0.1,
                               num_iterations: int = 100,
                               convergence_threshold: float = 1e-3,
                               random_reset_dev: float = 10e-3, # random noise around the center of parameter range to reset to when a parameter hits the limit
                               ):
    # Initialize history with NaN values to indicate uninitialized entries
    history = np.full(num_iterations + 1, np.nan)
    current_delta = init_delta.copy()
    history[0] = np.linalg.norm(current_delta)  # Record initial delta magnitude

    params = hardware_mesh.get_params()
    target_matrix = hardware_mesh.get() + init_delta

    reset_mean = 0.5 * (hardware_mesh.param_limits[1] - hardware_mesh.param_limits[0]) + hardware_mesh.param_limits[0]

    for iteration in range(num_iterations):
        gradients = np.zeros(hardware_mesh.num_params)
        for param_index in range(hardware_mesh.num_params):
            # Central difference approximation
            params_plus = params.copy()
            params_plus[param_index] += delta_voltage
            hardware_mesh.set_params(params_plus)
            meas_plus = hardware_mesh.measure_matrix()
            delta_plus = meas_plus

            params_minus = params.copy()
            params_minus[param_index] -= delta_voltage
            hardware_mesh.set_params(params_minus)
            meas_minus = hardware_mesh.measure_matrix()
            delta_minus = meas_minus

            # Gradient calculation
            grad_matrix = (delta_plus - delta_minus) / (2 * delta_voltage)
            grad_mag = -2*np.sum(grad_matrix * current_delta)  # Chain rule for Frobenius norm squared (element-wise product and sum)
            gradients[param_index] = grad_mag

        # Update parameter
        params -= learning_rate * gradients
        overflow_params = params > hardware_mesh.param_limits[1]
        params[overflow_params] = np.random.normal(loc=reset_mean, scale=random_reset_dev, size=np.sum(overflow_params)) # Prevent overflow
        hardware_mesh.set_params(params)

        # Measure new matrix and error
        new_meas = hardware_mesh.measure_matrix()
        new_delta = target_matrix - new_meas
        new_delta_mag = np.linalg.norm(new_delta)
        history[iteration + 1] = new_delta_mag
        current_delta = new_delta.copy()

        if new_delta_mag < convergence_threshold:  # Early stopping threshold
            break

    history = history[~np.isnan(history)]  # Remove uninitialized entries
    return history, params