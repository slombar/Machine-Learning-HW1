import numpy as np
import math


def problem1(A, B):
    return A + B

    '''Given matrices A, B, and C, compute and return AB − C (i.e., right-multiply matrix A by matrix
    B, and then subtract C). Use dot or np.dot
    '''
def problem2(A, B, C):
    return (np.dot(A, B)) - C

    '''
     Given matrices A, B, and C, return A (dot) B + C>, where (dot) represents the element-wise (Hadamard)
    product and > represents matrix transpose. In numpy, the element-wise product is obtained simply
    with *.
    '''
def problem3(A, B, C):
   return (np.array(A) * np.array(B)) + np.transpose(C)

'''
Given column vectors x and y and square matrix S, compute x>Sy.
'''
def problem4(x, S, y):
    return np.dot(np.dot(np.transpose(x), S), y)

'''
Given matrix A, return a vector with the same number of rows as A but that contains all ones. Use
np.ones
'''
def problem5(A):
    return np.ones(A.shape)

'''
Given matrix A, return a matrix with the same shape and contents as A except that the diagonal
terms (Aii for every valid i) are all zero.
'''
def problem6(A):
    #fill diagonal : array, value, wrap? (defalut false)
    np.fill_diagonal(A, 0)
    #fill opposite diagonal
    np.fill_diagonal(np.fliplr(A), 0)
    return A


'''
Given square matrix A and (scalar) α, compute A + αI, where I is the identity matrix with the same
dimensions as A. Use np.eye. 
'''
def problem7(A, alpha):
    return A + (alpha * np.eye(A.shape[0]))


'''
Given matrix A and integers i, j, return the ith column of the jth row of A, i.e., Aji. 
'''
def problem8(A, i, j):
    return A[i][j]

'''
Given matrix A and integer i, return the sum of all the entries in the ith row, i.e., Ej Aij . 
Do not use a loop, which in Python is very slow. Instead use the np.sum function.
'''
def problem9(A, i):
    return np.sum(A[i, :])


'''
Given matrix A and scalars c, d, compute the arithmetic mean (you can use np.mean) over all entries
of A that are between c and d (inclusive).
'''
def problem10(A, c, d):
    return np.mean(A[np.nonzero(np.logical_and(A >= c, A <= d))])

'''
Given an (n × n) matrix A and integer k, return an (n × k) matrix containing the right-eigenvectors
of A corresponding to the k eigenvalues of A with the largest magnitude. Use np.linalg.eig to
compute eigenvectors.
'''
def problem11(A, k):
    w, v = np.linalg.eig(A)
    w = np.argsort(w)
    w = np.flip(w)

    return v[:, w[:k]]

'''
Given square matrix A and column vector x, use np.linalg.solve to compute A−1x. Do not use
np.linalg.inv or ** -1 to compute the inverse explicitly; this is numerically unstable and can, in
some situations, give incorrect results. 
'''
def problem12(A, x):
    return np.linalg.solve(A, x)

'''
Given an n-vector x and a non-negative integer k, return a n × k matrix consisting of k copies of x.
You can use numpy methods such as np.newaxis, np.atleast 2d, and/or np.repeat.
'''
def problem13(x, k):
    return x[None, :] * np.ones((k,))[:, None]

'''
Given a matrix A with n rows, return a matrix that results from randomly permuting the rows (but not the columns) in A. 
Do not modify the input array A (use np.random.permutation) 
'''
def problem14(A):
    m = A
    return np.apply_along_axis(np.random.permutation, 0, m)


if __name__ == '__main__':

    #defaults
    A= np.array([
        [6, 3],
        [7, 1]
    ])

    B = np.array([
        [8, 2],
        [1, 20]
    ])

    C = np.array([
        [1, 2],
        [3, 4]
    ])

    S = np.array([
        [1, 2],
        [3, 4]
    ])

    #first problem matrices
    A1 = np.array([
        [1, 2, 3],
        [3, 4, 5]
    ])

    B1 = np.array([
    [2, 2, 2],
    [2, 2, 2]
    ])

    #2nd prob
    A2= np.array([
        [6, 3],
        [7, 1]
    ])

    B2 = np.array([
        [24, 2, 0],
        [1, 20, 8]
    ])

    C2 = np.array([
        [1, 2, 3],
        [3, 4, 5]
    ])

    #p6
    A6 = np.array([
        [1, 7, 3, 4, 5],
        [3, 4, 5, 5, 6],
        [6, 7, 8, 7, 9],
        [9, 3, 4, 8, 9],
        [1, 2, 5, 8, 9]
    ])

    #p8 & 9
    A8 = np.array([
        [12, 4, 6, 8],
        [3, 6, 9, 11],
        [1, 2, 13, 4]
    ])

    #p12
    A12 = np.array([
        [6, 3],
        [7, 1]
    ])
    x12 = A12[:, 1]

    #p13

    x13 = np.array([
        [1,2,3,4]
    ])

    #p14
    A14 = np.array([
        [2, 4, 6],
        [3, 6, 9],
        [1, 2, 1]
    ])

    x = 2
    y= 3
    i=2
    j=2
    alpha=3
    c=2
    d=3
    k=5

    result = problem1(A1,B1)
    print("Problem 1: \n" + str(result))

    result = problem2(A2,B2,C2)
    print("Problem 2: \n" + str(result))

    result = problem3(A,B,C)
    print("Problem 3: \n" + str(result))

    result = problem4(x,S,y)
    print("Problem 4: \n" + str(result))

    result = problem5(A)
    print("Problem 5: \n" + str(result))

    result = problem6(A6)
    print("Problem 6: \n" + str(result))

    result = problem7(A,alpha)
    print("Problem 7: \n" + str(result))

    result = problem8(A8,i,j)
    print("Problem 8: \n" + str(result))

    result = problem9(A8,i)
    print("Problem 9: \n" + str(result))

    result = problem10(A,c,d)
    print("Problem 10: " + str(result))

    result = problem11(A,k)
    print("Problem 11: \n" + str(result))

    result = problem12(A12,x12)
    print("Problem 12: \n" + str(result))

    result = problem13(x13,k)
    print("Problem 13: \n" + str(result))

    result = problem14(A14)
    print("Problem 14: \n" + str(result))