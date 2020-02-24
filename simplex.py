#!/usr/bin/env python3

# 553.762 Homework 3
# Copyright: Desh Raj (Johns Hopkins University)
# Apache 2.0

# This code simulates the simplex algorithm for finding
# the optimal solution to a linear programming problem,
# min c^T x, subject to Ax = b, x >= 0.
# It takes as input a matrix A, and vectors b and c,
# and outputs intermediate basic feasible solution 
# tableaus, and a final optimal solution, if it exists.
# There are two modes: if the columns for the initial
# basis are provided, it starts with that basis, 
# otherwise, it performs the big-M method to obtain
# the pretableau.

# Input format for files containing A, b, and c:
# Matrix A (mxn): m lines, each containing n floating
# point values separated by a blank space
# Vector b: single line containing m floats
# Vector c: single line containing n floats

# To write the output, supply a filename at a path
# that exists, otherwise an exception is raised.

import argparse, sys, random
import numpy as np
from tabulate import tabulate

def get_args():
	parser = argparse.ArgumentParser(description="Run the simplex algorithm on given problem." 
		"Usage: simplex.py [options...] <matrix-A-file> <vector-B-file> <vector-C-file> <out-file>"
		"E.g.: simplex.py --initial-basis data/basis --random-seed 1 data/A data/b data/c exp/out",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--initial-basis", type=str, dest = "initial_basis_file", default=None,
		help="File containing intial basis columns")

	parser.add_argument("--basis-index", type=int, dest="basis_index", default=1,
		help="0/1 indexing used for basis")

	parser.add_argument("--random-seed", type=int, dest="random_seed", default=0, 
		help="Seed to be used for randomization")

	parser.add_argument("matrix_A_file", help="File containing matrix A")
	parser.add_argument("vector_b_file", help="File containing vector b")
	parser.add_argument("vector_c_file", help="File containing vector c")
	parser.add_argument("output_file", help="File to write all outputs in")

	print(' '.join(sys.argv))

	args = parser.parse_args()

	return args

def read_args(args):
	A = np.loadtxt(args.matrix_A_file)
	b = np.loadtxt(args.vector_b_file)
	c = np.loadtxt(args.vector_c_file)

	basis = None
	if args.initial_basis_file is not None:
		basis = np.loadtxt(args.initial_basis_file, dtype=np.int)
		basis -= args.basis_index

	if not (A.ndim == 2 and b.ndim == 1 and c.ndim == 1):
		raise Exception("A must be matrix and b,c must be vectors") 
	if not (A.shape[0] == b.shape[0] and A.shape[1] == c.shape[0]):
		raise Exception("Incompatible sizes of A, b, and c")
	if basis is not None and basis.shape[0] != A.shape[0]:
		raise Exception("Provided basis does not have the same rank as A")
	
	fout = open(args.output_file, 'w')

	return A, b, c, basis, fout

# This is a helper function that takes the matrix A
# and vector b and row reduces them according to
# some row,column value, i.e., it makes the row,col
# element of A 1 and all other elements in that col 0.
def row_reduce(A,b,c,z,row,col):
	b[row] /= A[row][col]
	A[row] /= A[row][col]
	for i,cur_row in enumerate(A):
		if (i==row):
			continue
		b[i] -= b[row]*cur_row[col]
		cur_row -= A[row]*cur_row[col]
		z -= b[row]*c[col]
		c -= A[row]*c[col]
	return z

# This method computes an initial BFS given basis columns
def compute_initial_bfs(A,b,c,z,basis):
	print ("Computing initial BFS using columns", basis)
	for i,col in enumerate(basis):
		z = row_reduce(A,b,c,z,i,col)
	return z

# If initial basis is not provided, this method adds
# artificial variables with a large cost and uses them
# as basis.
def compute_initial_bfs_bigM(A,b,c,z,M):
	print ("No starting basis provided. Using big M method with M = {}".format(M))
	m,n = A.shape
	A_prime = np.identity(m)
	A = np.hstack((A, A_prime))
	basis = np.array(range(n,n+m))
	c_prime = -M*np.ones(m)
	c = np.concatenate([c, c_prime])
	return A, c, basis

# This method runs one step of the simplex method
def run_simplex_step(A,b,c,z,basis):
	pivot_col = np.argmax(c)
	ratio = b/A[:,pivot_col]
	valid_idx = np.where(A[:,pivot_col] > 0)[0]
	if (valid_idx.size == 0):
		return z, False
	pivot_row = valid_idx[ratio[valid_idx].argmin()]
	basis[pivot_row] = pivot_col
	z = row_reduce(A,b,c,z,pivot_row,pivot_col)
	return z, True

def print_tableau(A, b, c, z, fout):
	A = np.insert(A, 0, c.T, axis=0)
	b = np.insert(b, 0, z, axis=0)
	A = np.insert(A, 0, np.zeros(A.shape[0]), axis=1)
	A[0,0] = 1
	A = np.insert(A, A.shape[1], b, axis=1)
	table = tabulate(A, tablefmt="fancy_grid", floatfmt=".2f")
	print(table, file=fout)
	return

def compute_bfs(b,c,basis):
	x = np.zeros_like(c)
	for i,val in enumerate(basis):
		x[val] = b[i]
	return x

def simplex(A,b,c,z,basis,fout):
	step = 0
	isBounded = True
	bfs = compute_bfs(b,c,basis)

	while ((c>0).any()):
		step += 1
		print ("Simplex step {}; basis is {}".format(step, basis))
		print ("Simplex step {}; basis is {}".format(step, basis), file=fout)
		z, isBounded = run_simplex_step(A, b, c, z, basis)
		if not isBounded:
			print("The LP is unbounded", file=fout)
			print("Terminating")
			sys.exit()
		bfs = compute_bfs(b,c,basis)
		print_tableau(A.copy(), b.copy(), c, z, fout)

	print ("Finished in {} steps".format(step))

	return z, bfs

def main():
	args = get_args()
	A,b,c,basis,fout = read_args(args)

	# z is the objective function value we are minimizing
	z = 0
	c *= -1
	print ("Pretableau:", file=fout)
	print_tableau(A.copy(), b.copy(), c, z, fout)

	if basis is None:
		initialBasisProvided = False
		random.seed(args.random_seed)
		M = random.getrandbits(16)
		A, c, basis = compute_initial_bfs_bigM(A,b,c,z,M)
	else:
		initialBasisProvided = True
		z = compute_initial_bfs(A,b,c,z,basis)

	print ("Initial BFS:", file=fout)
	print_tableau(A.copy(), b.copy(), c, z, fout)

	z, bfs = simplex(A,b,c,z,basis,fout)
	if not initialBasisProvided:
		bfs = bfs[:-A.shape[0]]
	print("Optimal solution: {}".format(bfs), file=fout)
	print("Optimal objective function value = {:.2f}".format(z), file=fout)

	return

if __name__=="__main__":
	main()
