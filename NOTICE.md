# Notice: scope of the license and attribution

quantile_regression_rust: a Rust translation of rq.m, rqic.m and ipqr.m, and Rust replication code for
Arellano and Bonhomme (2017)

Copyright (c) 2026 Eliot R. Weinstein  
eliotw@uchicago.edu  
https://erweinstein.com

The Rust code is licensed under the BSD 3-Clause License (see `LICENSE`).

That license covers only the material written by Eliot R. Weinstein in this repository: the Rust code, the
tests, the reference script and the documentation. It does not cover, and grants no rights in, the following,
which remain the work and property of their authors and are subject to their own terms:

- The Frisch–Newton interior point algorithm for quantile regression and its MATLAB implementations rq.m and
  rqic.m: lp_fnm by Daniel Morillo and Roger Koenker, translated from Ox to MATLAB by Paul Eilers (1999),
  modified by Roger Koenker (2000–) and Paul Eilers (2004); rqic.m by Roger Koenker (2002). See
  http://www.econ.uiuc.edu/~roger/research/rq/rq.html and Portnoy, S. and R. Koenker (1997), *Statistical
  Science* 12(4), 279–300. (Koenker's R package quantreg, which implements the same algorithms, is licensed
  under the GNU GPL.)

- The nonlinear interior point algorithm and ipqr.m: Koenker, R. and B. J. Park (1996), *Journal of
  Econometrics* 71, 265–283.

- Selection-corrected quantile regression and the MATLAB code Code_QRS.m, by Manuel Arellano and Stéphane
  Bonhomme: Arellano, M. and S. Bonhomme (2017), "Quantile Selection Models With an Application to
  Understanding Changes in Wage Inequality," *Econometrica* 85(1), 1–28.

This translation follows the Python port in erweinstein/quantile_regression, which is unchanged.

If you use this code, please cite the original authors above.
