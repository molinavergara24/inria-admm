ADMM for Frictional Contact
=======

## Introduction

Implementaation of friction model as a parametric quadratic optimization problem with second-order cone constraints coupled with a fixed point equation. _See paper in this link_ [link](https://hal.inria.fr/inria-00495734)

---

## Table of Contents
- [Ballsfalling_oldcode](https://github.com/molinavergara24/inria-admm#ballsfalling_oldcode)
- [General code](https://github.com/molinavergara24/inria-admm#general-code)
  - [Time_comparison](https://github.com/molinavergara24/inria-admm#time_comparison)
  - [Varying_penalty](https://github.com/molinavergara24/inria-admm#varying_penalty)
    - [Hager](https://github.com/molinavergara24/inria-admm#hager)
    - [Spectral](https://github.com/molinavergara24/inria-admm#spectral)
    - [Wohlberg](https://github.com/molinavergara24/inria-admm#spectral)
- [Import data](https://github.com/molinavergara24/inria-admm#import-data)

---

## Ballsfalling_oldcode
Old code implemented for three balls falling in a vertical line.

## General code
Code implemented for data import (see [Import data]).

An optimal penalty parameter selection for QP has been used. _See paper in this link_ [link](https://arxiv.org/abs/1306.2454).

### Time_comparison
Comparison between simple, relaxed and fast ADMM. For general information see:
[link](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)
[link](http://epubs.siam.org/doi/abs/10.1137/120896219)

## Varying_penalty
Comparison between different approaches for varying penalty parameter.

### Hager
_See paper in this link_ [link](https://link.springer.com/article/10.1023/A:1004603514434)

### Spectral
_See paper in this link_ [link](https://arxiv.org/abs/1605.07246)

### Wohlberg
_See paper in this link_ [link](https://arxiv.org/abs/1704.06209)

## Import data
Examples which provide the variables that are needed. _See paper in this link_ [link](https://hal.inria.fr/hal-00782128)
