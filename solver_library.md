# TODO

Legend:
- GD = Gradient Descent
- N = Newton
- B = BFGS
- LB = L-BFGS
- LS = Line search (Armijo / Wolfe / Goldstein)
- TR = Trust-region
- QN = Quasi-Newton (BFGS / L-BFGS)

---

# Solver Library Roadmap (Implementation Order)
---

## Line Search Methods

- [x] Fixed step
- [x] Armijo
- [x] Wolfe (weak / strong)
- [x] Goldstein
- [x] Try-full-step wrapper
- [ ] Barzilai-Borwein spectral step policy
- [ ] Backtracking with quadratic/cubic interpolation
- [ ] Exact line search

---
## Phase 0: Core Solver Platform

- [x] Solver API (`obj, x0, Options, callbacks, step strategy`)
- [x] Trace output
- [x] Oracle eval accounting
- [x] FD gradient + Hessian fallback
- [x] Step attempt/status plumbing (`eval_failed`, `line_search_failed`, `max_evals`)
- [x] Single + multi-cache support in Oracle (option-controlled)
- [x] Hessian-vector products API integration
- [x] Curvature diagnostics
- [x] Condition number estimates
- [ ] High-dimensional benchmark harness
- [ ] Parallelism/vectorization
- [ ] Trace observer/export
- [ ] Plotting

## Phase 1: Unconstrained First-Order Expansion

- [x] Gradient Descent
- [ ] Nonlinear Conjugate Gradient (Fletcher-Reeves)
- [ ] Nonlinear Conjugate Gradient (Polak-Ribiere+)
- [ ] Nonlinear Conjugate Gradient (Hestenes-Stiefel)
- [ ] Nonlinear Conjugate Gradient (Dai-Yuan)
- [ ] Nonlinear Conjugate Gradient (Hager-Zhang)
- [ ] Restarted nonlinear CG
- [ ] Preconditioned nonlinear CG
- [ ] Spectral gradient methods (unconstrained core; projected variants in constrained phase)

## Phase 2: Unconstrained Second-Order / Quasi-Newton Expansion

- [x] Newton/Damped Newton
- [x] BFGS
- [x] L-BFGS
- [ ] Levenberg-Marquardt (LM)
- [x] L-BFGS
- [x] DFP
- [ ] SR1
- [ ] PSB

## Phase 3: Trust-Region / Large-Scale Direction

- [ ] Trust-region scaffold (options + abstraction hooks)
- [ ] Trust-region Gradient Descent
- [ ] Trust-region Newton
- [ ] Trust-region BFGS
- [ ] Dogleg
- [ ] Double-dogleg
- [ ] Truncated-CG trust-region subproblem solver (Steihaug)
- [ ] Hessian-Free Newton (Hv via FD)
- [ ] Newton-CG
- [ ] Inexact Newton
- [ ] Gauss-Newton
- [ ] Gauss-Newton + LM damping

## Phase 4: Stochastic / Derivative-Free (Unconstrained)

- [ ] SGD
- [ ] Mini-batch SGD
- [ ] Momentum
- [ ] Nesterov Accelerated Gradient
- [ ] AdaGrad
- [ ] RMSProp
- [ ] Adam
- [ ] AdamW
- [ ] Coordinate Descent
- [ ] Random Coordinate Descent
- [ ] Block Coordinate Descent
- [ ] Pattern Search
- [ ] Nelder-Mead
- [ ] Powell's Method
- [ ] Hooke-Jeeves

## Phase 5: Constrained Optimization Stack

### Penalty / Barrier

- [ ] Quadratic penalty
- [ ] Exact penalty (L1)
- [ ] Augmented Lagrangian (ALM)
- [ ] Log-barrier
- [ ] Interior penalty

### Interior-Point

- [ ] Primal interior-point
- [ ] Dual interior-point
- [ ] Primal-dual interior-point
- [ ] Mehrotra predictor-corrector IPM

### Sequential Methods

- [ ] Basic SQP (exact Hessian)
- [ ] SQP with BFGS
- [ ] SQP with L-BFGS
- [ ] Trust-region SQP
- [ ] Filter-SQP
- [ ] SLP with trust regions

### Active-Set / Projection / KKT

- [ ] Active-set QP
- [ ] Active-set NLP
- [ ] Bound-constrained active-set
- [ ] Projected gradient
- [ ] Projected Newton
- [ ] Projected BFGS
- [ ] Projected L-BFGS
- [ ] Alternating projections
- [ ] KKT Newton
- [ ] Reduced-space methods
- [ ] Null-space methods

### Trust-Region Constrained

- [ ] Trust-region constrained Newton
- [ ] Trust-region interior-point
- [ ] Trust-region SQP

### Specialized Constraint Types

- [ ] L-BFGS-B
- [ ] Projected Newton for box constraints
- [ ] Gradient projection method (projected gradient baseline)
- [ ] Active-set QP solver (linear constraints)
- [ ] Interior-point QP
- [ ] Semidefinite programming (SDP) for Linear Matrix Inequalities (LMIs): primal-dual interior-point baseline

### Nonsmooth / Composite

- [ ] Proximal gradient
- [ ] Proximal Newton
- [ ] ADMM
- [ ] Douglas-Rachford splitting

### Global / Heuristic

- [ ] Genetic algorithms with constraints
- [ ] Penalty-based evolutionary methods

## Phase 6: Additional First-Order / Composite Families

- [ ] Heavy-ball momentum solver variant
- [ ] Nesterov accelerated gradient variant (with restart safeguards)
- [ ] Mirror descent (Euclidean and entropy mirror maps)
- [ ] Conditional-gradient / Frank-Wolfe family (with dual-gap diagnostics)
- [ ] Proximal-point baseline
- [ ] Accelerated proximal variants beyond FISTA
- [ ] Anderson acceleration wrapper for fixed-point/proximal iterations

## Phase 7: Advanced Constrained / Saddle-Point Methods

- [ ] PDHG / Chambolle-Pock primal-dual splitting
- [ ] Semismooth-Newton / primal-dual active-set method
- [ ] Exact-penalty + smoothing continuation workflow
- [ ] Stabilized SQP variants (elastic mode / restoration phase)
- [ ] ALADIN-style distributed augmented-Lagrangian sequential method

## Phase 8: Global / Derivative-Free Expansion

- [ ] CMA-ES baseline
- [ ] Differential Evolution baseline
- [ ] Simulated Annealing baseline
- [ ] Basin-Hopping local-global hybrid
- [ ] Model-based DFO methods (COBYLA/BOBYQA-style trust-region interpolation)
- [ ] Global-search benchmark suite for noisy/nonconvex black-box objectives

## Phase 9: Large-Scale and Stochastic Second-Order

- [ ] Stochastic L-BFGS / online quasi-Newton
- [ ] Subsampled Newton
- [ ] Sketchy Newton
- [ ] Natural-gradient approximation hooks (K-FAC/Shampoo-style adapters)
- [ ] Variance-reduced first-order methods (SVRG / SAGA / SARAH)

## Phase 10: Specialized Problem Classes and Interfaces

- [ ] Robust nonlinear least-squares losses (Huber / Cauchy / Tukey)
- [ ] Conic/structured QP interface mode (active-set and operator-splitting backend adapters)
- [ ] Conic/SDP interface mode for LMIs (PSD cone constraints, SDP data model, solver adapter layer)
- [ ] Min-max / saddle-point methods (extragradient / optimistic gradient)
- [ ] Multiobjective optimization baseline (Pareto-front sampling + weighted-sum continuation)
- [ ] Mixed-integer outer-loop strategy hook (branch-and-bound with continuous relaxations)

---

# Full Solver Catalog (Family View)

Most of these are from:
- @nocedal2006numerical
- https://optimization.cbe.cornell.edu/index.php?title=Main_Page
- https://indrag49.github.io/Numerical-Optimization/

## Unconstrained: First-Order

- [ ] Gradient Descent
- [ ] Spectral gradient (unconstrained)
- [ ] Nonlinear CG (FR)
- [ ] Nonlinear CG (PR+)
- [ ] Nonlinear CG (HS)
- [ ] Nonlinear CG (DY)
- [ ] Nonlinear CG (HZ)
- [ ] Restarted nonlinear CG
- [ ] Preconditioned nonlinear CG
- [ ] Barzilai-Borwein (spectral step policy)

## Unconstrained: Second-Order / QN

- [x] Newton/Damped Newton (Modified)
- [ ] Levenberg-Marquardt
- [x] BFGS
- [x] L-BFGS
- [x] DFP
- [ ] SR1
- [ ] PSB

## Unconstrained: Trust-Region / Large-Scale

- [ ] Trust-region built-ins (Cauchy / Dogleg / Steihaug)
- [ ] Trust-region GD
- [ ] Trust-region Newton
- [ ] Trust-region BFGS
- [ ] Dogleg
- [ ] Double-dogleg
- [ ] Truncated-CG trust-region solver (Steihaug)
- [ ] Hessian-Free Newton
- [ ] Newton-CG
- [ ] Inexact Newton
- [ ] Gauss-Newton
- [ ] Gauss-Newton + LM

## Unconstrained: Stochastic / Derivative-Free

- [ ] SGD / Mini-batch SGD
- [ ] Momentum / NAG
- [ ] AdaGrad / RMSProp / Adam / AdamW
- [ ] Coordinate Descent family
- [ ] Pattern Search / Nelder-Mead / Powell / Hooke-Jeeves

## Unconstrained: Additional First-Order / Composite

- [ ] Heavy-ball momentum
- [ ] Nesterov accelerated gradient with restart safeguards
- [ ] Mirror descent (Euclidean / entropy maps)
- [ ] Conditional-gradient / Frank-Wolfe
- [ ] Proximal-point method
- [ ] Accelerated proximal variants beyond FISTA
- [ ] Anderson acceleration wrapper (fixed-point/proximal iterations)

## Constrained Families

- [ ] Penalty / Barrier
- [ ] Interior-Point
- [ ] SQP / SLP
- [ ] Active-Set
- [ ] Projection methods
- [ ] KKT-space methods
- [ ] Trust-region constrained methods
- [ ] Specialized bound/linear methods (L-BFGS-B, QP variants)
- [ ] Nonsmooth composite (Prox, ADMM, DR)
- [ ] Global constrained heuristics

## Constrained: Advanced Primal-Dual / Saddle-Point

- [ ] PDHG / Chambolle-Pock
- [ ] Semismooth-Newton / primal-dual active-set
- [ ] Exact-penalty + smoothing continuation
- [ ] Stabilized SQP (elastic / restoration)
- [ ] ALADIN-style distributed augmented-Lagrangian sequential method

## Global / Derivative-Free: Advanced

- [ ] CMA-ES
- [ ] Differential Evolution
- [ ] Simulated Annealing
- [ ] Basin-Hopping
- [ ] Model-based DFO (COBYLA/BOBYQA-style)

## Large-Scale / Stochastic Second-Order

- [ ] Stochastic L-BFGS / online quasi-Newton
- [ ] Subsampled Newton
- [ ] Sketchy Newton
- [ ] Natural-gradient adapters (K-FAC / Shampoo style)
- [ ] Variance-reduced first-order family (SVRG / SAGA / SARAH)

## Specialized Problem Classes / Interfaces

- [ ] Robust nonlinear least-squares losses (Huber / Cauchy / Tukey)
- [ ] Conic / structured QP interface mode
- [ ] Min-max / saddle-point methods (extragradient / optimistic gradient)
- [ ] Multiobjective optimization baseline
- [ ] Mixed-integer outer-loop strategy hook

