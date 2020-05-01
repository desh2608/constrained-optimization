#!/usr/bin/env python3

# 553.762 Homework 6
# Copyright: Desh Raj (Johns Hopkins University)
# Apache 2.0

# This code simulates the long-step interior point
# algorithm for quadratic programs.
# Input format for files containing Q, c, A, and b:
# Matrix Q (nxn): symmetric, positive definite
# Vector c: single line containing n floats
# Matrix A (mxn): m lines, each containing n floating
# point values separated by a blank space
# Vector b: single line containing m floats

import argparse, sys, random
import numpy as np
import scipy.optimize as sopt

LR=0.01

def get_args():
    parser = argparse.ArgumentParser(description="Run the penalty method on give problem. The\n"
        " auxiliary function is solved using steepest descent, terminating when update becomes\n"
        " smaller than the threshold."
        " Usage: penalty.py [options...]\n"
        " E.g.: penalty.py --beta 2 --gamma 2 --tol 0.001",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--delta", type=float, dest = "delta", default=2.0,
        help="Initial positive scalar multiplier for the constraint terms in the auxiliary function.\n"
            " This is doubled every iteration.")
    parser.add_argument("--gamma", type=int, dest = "gamma", default=2,
        help="Positive integer exponent for the constraints in the auxiliary function.")
    parser.add_argument("--tol", type=float, dest="tol", default=0.001,
        help="Tolerance for terminating penalty method")
    parser.add_argument("--random-seed", type=int, dest="random_seed", default=0, 
        help="Seed to be used for randomization")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    return args

def check_args(args):
    if not (args.delta > 1):
        raise Exception("Beta must be a positive scalar value greater than 1")
    if not (args.gamma > 0):
        raise Exception("Gamma must be a positive integer")
    return


# Define the objective function and the constraints here
################################################################
def f(x):
    return np.exp(-1.0*(np.sum(x)))

def df(x):
    return (-1*f(x)*np.ones_like(x))

def g1(x):
    y = np.power(x,2)
    return (2*y[0] + y[1] - 1)

def dg1(x):
    return np.array([4*x[0], 2*x[1]])

def g2(x):
    return (x[0] - 0.5)

def dg2(x):
    return np.array([1,0])
###############################################################

# Helper functions
def relu(x):
    return np.maximum(0,x)

def relu_grad(x):
    return int(x>0)

def alpha(x, gamma):
    return (np.power(relu(g1(x)),gamma) + np.power(relu(g2(x)),gamma))

def dalpha(x, gamma):
    grad1 = gamma*(np.power(relu(g1(x)),gamma-1))*dg1(x)*relu_grad(g1(x))
    grad2 = gamma*(np.power(relu(g2(x)),gamma-1))*dg2(x)*relu_grad(g2(x))
    return np.add(grad1,grad2)

def Psi(x, beta, gamma):
    return (f(x) + beta*alpha(x, gamma))

def dPsi(x, beta, gamma):
    grad = df(x) + beta*dalpha(x,gamma)
    return grad

def steepest_descent(x, beta, gamma, theta=0.001):
    while True:
        dx = dPsi(x, beta, gamma)
        if (Psi(x, beta, gamma) - Psi(x - LR*dx, beta, gamma) < theta):
            break
        x = x - LR*dx
    return x

# This function performs the penalty method
def penalty(delta, gamma, tol):
    x = np.array([0,0])
    k = 0
    while True:
        k += 1
        beta = delta**k
        x = steepest_descent(x, beta, gamma)
        print ("Step",k,"optimal x:",x)
        if (beta*alpha(x, gamma) < tol):
            break
    return x, k

def main():
    args = get_args()
    check_args(args)

    x, num_iter = penalty(args.delta, args.gamma, args.tol)

    print ("Optimal x is: {}".format(np.around(x, decimals=2)))
    print ("Optimal objective function value: {}".format(np.around(f(x), decimals=2)))
    print ("Number of iterations required: {}".format(num_iter))
    #print (args.delta,",",x,",",num_iter)
    
    return

if __name__=="__main__":
    main()
