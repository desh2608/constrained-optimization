## Homework 9

#### Author: Desh Raj (Johns Hopkins University)

This code implements the penalty method for the problem
e^(-x1-x2) s.t. 2*x1^2 + x2^2 <= 1, x1 <= 0.5.

For solving the auxiliary problem, we use steepest descent.
All implementation is done using `numpy`. Steepest descent
optimization for each step is done using `scipy.optimize.golden`.

Some outputs for different initial values of beta:

* beta = 2
```
Step 1 optimal x: [0.41475743 0.82951419]
Step 2 optimal x: [0.41475743 0.82951419]
Step 3 optimal x: [0.40838615 0.82314291]
Optimal x is: [0.41 0.82]
Optimal objective function value: 0.29
Number of iterations required: 3
```

* beta = 3
```
Step 1 optimal x: [0.41632752 0.82604557]
Step 2 optimal x: [0.41049824 0.82028555]
Optimal x is: [0.41 0.82]
Optimal objective function value: 0.29
Number of iterations required: 2
```

* beta = 1.5
```
Step 1 optimal x: [0.37757227 0.87915978]
Step 2 optimal x: [0.37757227 0.87915978]
Step 3 optimal x: [0.36924723 0.86820076]
Step 4 optimal x: [0.36924723 0.86820076]
Step 5 optimal x: [0.36542365 0.86306347]
Step 6 optimal x: [0.36542365 0.86306347]
Step 7 optimal x: [0.36542365 0.86306347]
Step 8 optimal x: [0.36312795 0.86015065]
Optimal x is: [0.36 0.86]
Optimal objective function value: 0.29
Number of iterations required: 8
```

* beta = 5
```
Step 1 optimal x: [0.37182527 0.86119118]
Step 2 optimal x: [0.36773547 0.85627706]
Optimal x is: [0.37 0.86]
Optimal objective function value: 0.29
Number of iterations required: 2
```
