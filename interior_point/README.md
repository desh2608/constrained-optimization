# Long-step interior point method

This is a simple Python implementation for simulating the 
long-step interior point method for solving quadratic
programming problems of the form:

min (x^T Q x) /2 + c^T x s.t. Ax = b, x >= 0

The code was written as part of a homework assignment
for course 553.762 (Constrained Optimization) during 
Spring 2020 term at Johns Hopkins University.

Author: Desh Raj (draj@cs.jhu.edu)

## How to run?

```
./interior_point.py [options..] <matrix-Q-file> <vector-c-file> <matrix-A-file> <vector-b-file>
```

The optional arguments are:
`--random-seed`: provide a random seed for choosing epsilon
`--delta`: delta value for the long step method. Smaller delta allows more 
deviation from central path. 
`--epsilon-min`: minimum multiplier value for adjusted Newton method
`--epsilon-max`: maximum multiplier value for adjusted Newton method

## Example usage

```
./interior_point.py --delta 0.5 sample/Q sample/c sample/A sample/b
```

