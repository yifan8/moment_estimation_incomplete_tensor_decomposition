## test RMETC

from solver.solver import RMETC
from util import sample_gauss, sample_poisson, sample_exp, sample_bernoulli, sample_gamma, match

import numpy as np
import scipy.linalg as la
from scipy.optimize import line_search
from qpsolvers import solve_ls 
import numpy.random as rand  
from scipy.special import comb
from scipy.stats import multivariate_normal as normal


# print options
np.set_printoptions(precision = 6, suppress=True)


# basic dimensionalities
n = 30
r = 18
p = 20000
d = 4
rmax = comb(np.floor((n - 1)/2), np.floor(d/2))

assert r <= rmax, "rank exceeds maximal rank"

truth_seed = 123
init_seed = 1234

test = 'gauss'

print()
print(f'test = {test}')
print(f'n = {n}, r = {r}, p = {p}')
print(f'truth seed = {truth_seed}, init seed = {init_seed}')


# generate random means, weights, and data
if truth_seed is not None:
    rand.seed(truth_seed)
    
rv = normal(cov = np.eye(n)) 

if test == 'gauss':
    ws = rand.uniform(1., 5., size = r)
    std = np.abs(rv.rvs(r).T)
    std[std < 0.01] = 0.01
    A = rv.rvs(r).T
    V, Ns, label, w, A, std = sample_gauss(A, ws, std, p, get_label = True)
    
if test == 'exp':
    ws = rand.uniform(low = 1, high = 5, size = r)
    loc = rv.rvs(r).T * 0
    scale = rand.uniform(low = 0, high = 1, size = (n, r))
    V, Ns, label, w, A, std = sample_exp(scale, ws, loc = loc, N = p, get_label = True)
    
if test == 'bernoulli':
    ws = rand.uniform(low = 1, high = 5, size = r)
    A = rand.uniform(low = 0, high = 1, size = (n, r))
    V, Ns, w, A, std = sample_bernoulli(A, ws, N = p)
    
if test == 'poisson':
    ws = rand.uniform(low = 1, high = 5, size = r)
    A = rand.uniform(low = 0, high = 10, size = (n, r))
    V, Ns, label, w, A, std = sample_poisson(A, ws, p, get_label = True)
    
if test == 'gamma':
    ws = rand.uniform(1., 5., size = r)
    shape = rand.uniform(low = 1, high = 5, size = (n, r))
    scale = rand.uniform(low = 0.1, high = 5, size = (n, r))
    V, Ns, w, A, std = sample_gamma(scale, shape, ws, p)


# true second moment, moment generating function at 0.5, -0.5
# NOTE that mgf at 0.5 or -0.5 may NOT exist for distributions like Gamma. 
# Thus the computed result is subject to a large error.
# Here we use Gaussian distributions as an illustration for mgf calculation

M2 = np.zeros((n, r))
mgf_p = np.zeros((n, r))
mgf_m = np.zeros((n, r))
start = 0
for i in range(r):
    curV = V[:, start : start + Ns[i]]
    start += Ns[i]
    M2[:, i] = np.mean((curV)**2, axis = 1)
    mgf_p[:, i] = np.mean(np.exp(0.5 * curV), axis = 1)
    mgf_m[:, i] = np.mean(np.exp(-0.5 * curV), axis = 1)


# initial guess
if init_seed is not None:
    rand.seed(init_seed)

A0 = normal(cov = np.eye(n)).rvs(r).T
w0 = np.ones(r) / r 


# setup optimizers 
def qp_solver(b, W, lb):
    return solve_ls(np.eye(r), b, A = np.ones(r), b = np.ones(1), lb = np.ones(r) * lb, W = W, solver = 'quadprog')

def std_solver(b, W, lb = None):
    return solve_ls(np.eye(r), b, lb = lb, W = W, solver = 'quadprog')

def wt_scheduler(METC):
    return 0.1 / METC.r

def nb_scheduler(METC):
    return -1 if not METC.warmup else METC.n // 2

def linsch(f, jac, x0, a, amax, fx0 = None, dfx0 = None):
    soln = line_search(f, jac, x0, a, amax, gfk = dfx0, old_fval = fx0)
    if soln[0] is None:
        return x0
    else:
        return x0 + soln[0] * a


# initialize solver
solver = RMETC(d)
solver.set_optimizer(qp_solver = qp_solver, wt_lb_scheduler = wt_scheduler, 
                     nb_scheduler = nb_scheduler, line_search_device = linsch)

Atol = 1E-4
wtol = 1E-4
ftol = None
maxiter = 200


# solve mean and weights
comp_A, comp_w, conv = \
    solver.solve_mean_weight(V, A0, w0, maxiter, Atol, wtol, ftol, 
                              translate_init = False, 
                              warmup = 20, AA_depth = 15, 
                              AAtol = 1E-4, monit = 1)


# solve second moments
comp_std = solver.solve_std(std_solver, reg = 1E-2)
comp_M2 = comp_std ** 2 + comp_A ** 2


# evaluate moment generating function at grid points
def mgf(x, grid):
    return np.exp(np.outer(x, grid))

comp_mgf_p, comp_mgf_m = solver.solve_gen_mean(mgf, k = 2, grid = [0.5, -0.5])


# match the computed results
comp_A, comp_w, comp_std, comp_M2, comp_mgf_p, comp_mgf_m = match(comp_A, A, comp_w, comp_std, comp_M2, comp_mgf_p, comp_mgf_m)


# print test results
print()
print(f'Full iteration: {solver.niter}')
print()
print("Relative error between weights")
print(la.norm(w - comp_w) / la.norm(w))
print()
print('Relative Frobenius error on mean')
print(la.norm(A - comp_A, 'fro') / la.norm(A, 'fro'))
print()
print('Relative Frobenius error on the second moment')
print(la.norm(M2 - comp_M2, 'fro') / la.norm(M2, 'fro'))
print()
print('Relative Frobenius error on the moment generating function at 0.5')
print(la.norm(comp_mgf_p - mgf_p, 'fro') / la.norm(mgf_p, 'fro'))
print()
print('Relative Frobenius error the moment generating function at -0.5')
print(la.norm(comp_mgf_m - mgf_m, 'fro') / la.norm(mgf_m, 'fro'))
print()