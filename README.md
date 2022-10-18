# About
This is an implementation of the algorithms in Moment Estimation for Nonparametric Mixture Models through Implicit Tensor Decomposition by Yifan Zhang and Joe Kileel. The name (METC) stands for Moment Estimation through Tensor Completion.

## Required Python Package:
* Numpy
* Scipy

## API:
An outline of the API is provided below. For details, please refer to the docstrings in the main code (./solver/solver.py). A demo can be found in demo.py (requires additional packages to run).

### solve_mean_weight
Solve the means and weights using the $\text{ALS}^{++}$ algorithm.

### solve_std
Solve the standard deviation of each class after solving for the means and weights.

### solve_gen_mean
Solve the general means $\mathbb{E}_{X\sim\mathcal{D_j}}[g(X)]$ for coordinatewise functions $g$ after solving the means and weights.

### get_soln
Extract the means and weights computed by solve_mean_weight

### get_data
Extract the data from the solver

### eval_cost
Evaluate the cost at given means and weights

### eval_deriv
Evaluate the derivatives at given means and weights

### update_params
Update the current means and weights stored in the solver to the given ones

### set_optimizer
Config subroutine solvers like quadratic programming or line search used in the $\text{ALS}^{++}$ algorithm.

### norm_tau_weights
Compute the normalized $\tau$ weights $\tau_i = \frac{(n-i)!}{n!}$.
