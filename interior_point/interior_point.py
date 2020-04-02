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

def get_args():
    parser = argparse.ArgumentParser(description="Run the interior point algorithm on given problem." 
        "Usage: interior_point.py [options...] <matrix-Q-file> <vector-c-file> <matrix-A-file> <vector-b-file>"
        "E.g.: simplex.py --delta 0.1 --epsilon-min 0.2 --epsilon-max 0.6 --random-seed 1 Q c A b",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--delta", type=float, dest = "delta", default=0.5,
        help="Delta value for the long step method. Smaller delta allows more deviation from central path.")

    parser.add_argument("--epsilon-min", type=float, dest="epsilon_min", default=0.1,
        help="Minimum multiplier value for adjusted Newton method.")

    parser.add_argument("--epsilon-max", type=float, dest="epsilon_max", default=0.9,
        help="Maximum multiplier value for adjusted Newton method.")
    
    parser.add_argument("--random-seed", type=int, dest="random_seed", default=0, 
        help="Seed to be used for randomization")

    parser.add_argument("matrix_Q_file", help="File containing matrix Q")
    parser.add_argument("vector_c_file", help="File containing vector c")
    parser.add_argument("matrix_A_file", help="File containing matrix A")
    parser.add_argument("vector_b_file", help="File containing vector b")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    return args

def read_args(args):
    Q = np.loadtxt(args.matrix_Q_file)            
    c = np.loadtxt(args.vector_c_file)
    A = np.loadtxt(args.matrix_A_file)
    b = np.loadtxt(args.vector_b_file)

    if not (Q.ndim == 2 and A.ndim == 2 and b.ndim == 1 and c.ndim == 1):
        raise Exception("Q,A must be matrices and b,c must be vectors") 
    if not (A.shape[0] == b.shape[0] and A.shape[1] == c.shape[0]):
        raise Exception("Incompatible sizes of A, b, and c")
    if not (np.all(np.linalg.eigvals(Q) > 0)):
        raise Exception("Q must be positive definite")
    for x in [args.delta, args.epsilon_min, args.epsilon_max]:
        if not (x > 0 and x < 1):
            raise Exception("delta, epsilon_min, and epsilon_max must be between 0 and 1")

    return Q, c, A, b
    

def interior_point(Q,c,A,b,delta,epsilon_min,epsilon_max,h=0.00001):
    n = Q.shape[0]
    m = len(b)
    X = np.ones(n+m+n)
    
    def _get_sol(X):
        return X[:n], X[n:n+m], X[n+m:]

    def F(x,y,z):
        out = np.zeros(n+m+n)
        out[:n] = Q.dot(x) + c + A.T.dot(y) - z
        out[n:n+m] = A.dot(x) - b
        out[n+m:] = np.multiply(z,x)
        return out

    def F_tau(x,y,z,tau):
        out = F(x,y,z)
        out[n+m:] -= tau*np.ones_like(x)
        return out

    def grad_F_transpose(x,y,z):
        out = np.zeros((n+m+n,n+m+n))
        out[:n,:n] = Q
        out[:n,n:n+m] = A.T
        out[:n, n+m:] = -1*np.identity(n)
        out[n:n+m,:n] = A
        out[n+m:,:n] = np.diag(z).copy()
        out[n+m:,n+m:] = np.diag(x).copy()
        return out
    
    def binary_line_search(X, del_X, condition):
        start = 0
        end = 1
        while (end-start > h):
            alpha = (start + end)/2
            if (condition(X + alpha*del_X)):
                start = alpha
            else:
                end = alpha
        return alpha

    def long_step_neighborhood(X):
        x_cur, y_cur, z_cur = _get_sol(X)
        if (np.all(x_cur > 0) and np.all(z_cur > 0) and 
                np.all(np.multiply(z_cur,x_cur) - (delta*np.dot(z,x)/n)*np.ones(n) >= 0)):
            return True
        else:
            return False
    
    x,y,z = _get_sol(X)
    k = 0

    while np.any(F(x,y,z) >= 0.01):
        epsilon = random.uniform(epsilon_min, epsilon_max)
        beta = np.dot(z,x)/n
        del_X = np.linalg.solve(grad_F_transpose(x,y,z), -1*F_tau(x,y,z,epsilon*beta))
        alpha_k = binary_line_search(X, del_X, long_step_neighborhood)
        X += alpha_k * del_X
        x,y,z = _get_sol(X)
        del_x, del_y, del_z = _get_sol(del_X)
        k += 1

    return X[:n],X[n:n+m],X[n+m:],k

def main():
    args = get_args()
    Q,c,A,b = read_args(args)
    n = Q.shape[0]
    m = len(b)

    random.seed(args.random_seed)
    x_opt,y_opt,z_opt,k = interior_point(Q, c, A, b, args.delta, args.epsilon_min,
            args.epsilon_max)

    print ("Optimal x is: {}".format(np.around(x_opt, decimals=2)))
    print ("KKT multiplier y is: {}".format(np.around(y_opt, decimals=2)))
    print ("KKT multiplier z is: {}".format(np.around(z_opt, decimals=2)))
    print ("Number of iterations: {}".format(k))
    
    return

if __name__=="__main__":
    main()
