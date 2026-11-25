# Notice: scope of the license and attribution

quantile_regression_rust: a Rust translation of rq.m, rqic.m and ipqr.m, and Rust replication code for
Arellano and Bonhomme (2017)

Copyright (c) 2026 Eliot R. Weinstein  
eliotw@uchicago.edu  
https://erweinstein.com

The Rust code, including the Python bindings in `python/`, is licensed under either of

- the Apache License, Version 2.0 (`LICENSE-APACHE` or http://www.apache.org/licenses/LICENSE-2.0), or
- the MIT license (`LICENSE-MIT` or http://opensource.org/licenses/MIT),

at your option. (Until 25 November 2026 it was licensed under the BSD 3-Clause License, as the Python port
erweinstein/quantile_regression still is; copies obtained under that license remain under it.)

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

## Third-party code translated here

Two routines are translations of code under BSD-style licenses. Their notices apply to those parts, in addition
to the license above:

- `fminbound` in `src/ipqr.rs` translates the bounded Brent minimizer of SciPy 1.16.3
  (`scipy.optimize._optimize._minimize_scalar_bounded`), under SciPy's license:

  > Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
  > All rights reserved.
  >
  > Redistribution and use in source and binary forms, with or without modification, are permitted provided that
  > the following conditions are met:
  >
  > 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
  >    following disclaimer.
  > 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
  >    following disclaimer in the documentation and/or other materials provided with the distribution.
  > 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
  >    promote products derived from this software without specific prior written permission.
  >
  > THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
  > WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
  > PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
  > DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  > PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  > HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  > NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  > POSSIBILITY OF SUCH DAMAGE.

- The bivariate normal cdf in `src/normal.rs` (`Bvn`, `bvn_upper`, `bvn_cdf`) translates Alan Genz's Fortran
  routine BVND (tvpack.f; Genz 2004). Genz distributes his Fortran software under a BSD-style license with the
  same three conditions and disclaimer as above ("Copyright (C) Alan Genz, All rights reserved"; the exact
  wording and year are in the header of tvpack.f, at http://www.math.wsu.edu/faculty/genz/software/software.html).

If you use this code, please cite the original authors above.
