'''bootstrap_check.py -- the Python port's weighted bootstrap on the same data as checks/bootstrap_check.sh, for
comparing standard errors with the Rust version's (the draws differ: numpy's generator against Xoshiro256++).
Usage: python checks/bootstrap_check.py <path to erweinstein/quantile_regression> <data.csv> <reps>

Eliot R. Weinstein / eliotw@uchicago.edu / erweinstein.com
'''
import sys, time, warnings
import numpy as np
sys.path.insert(0, sys.argv[1])
from qr_extensions import selection_qr_bootstrap  # noqa: E402
warnings.simplefilter('ignore')
d = np.genfromtxt(sys.argv[2], delimiter=',', names=True)
t = time.time()
r = selection_qr_bootstrap(d['D'], d['Y'], d['X'], d['B'], reps=int(sys.argv[3]), seed=2017)
print(f"python: {time.time() - t:.1f} s; rho se {r['rho_se']:.4f}; beta se at tau = .5: {r['beta_se'][:, 9]}")
