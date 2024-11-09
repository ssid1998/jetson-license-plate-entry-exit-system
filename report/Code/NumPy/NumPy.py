"""! @brief Python script for Python package NumPy."""

############################
# Creator of the File: Vaishnavi Rane
# Date created: 30.1.2024
# Project: ML23-02-JetBot-Mapping-from-the-Floor
# Version: 2.0
# Reviewed by: Vaishnavi Rane
# Review Date: 05.02.2024
############################
#@mainpage NumPy Package
#@section intro_sec Introduction
#NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
# This script demonstrates basic usage of NumPy library for data manipulation and analysis.
#@section package NumPy_example
#
# - Modified and documented by Vaishnavi Rane on 30.1.2024.
#




# Best practice, use an environment rather than install in the base env
conda create -n my-env
conda activate my-env
# If you want to install from conda-forge
conda config --env --add channels conda-forge
# The actual install command
conda install numpy

pip install numpy


import numpy as np


# NumPy Version Check
import numpy 
print(numpy.__version__)

# NumPy Example Files
import numpy as np
arr = np.array([1, 2, 3, 4]) np.save('my_array', arr)
loaded_array = np.load('my_array.npy')

#Creating Array
# Creating a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])

# Creating a 2D array filled with zeros
arr_zeros = np.zeros((2, 3))

# Creating a 3x3 identity matrix
arr_identity = np.eye(3)

# Creating an array with evenly spaced values
arr_range = np.arange(1, 10, 2)

# Creating an array with linearly spaced values
arr_linspace = np.linspace(0, 1, 5)

#Array Operations
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Element-wise addition
result_add = arr1 + arr2

# Element-wise multiplication
result_mul = arr1 * arr2

# Matrix multiplication
result_dot = np.dot(arr1, arr2)

# Trigonometric functions
sin_arr = np.sin(arr1)

#Array Indexing and Slicing
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Accessing single element
element = arr[1, 2]  # Output: 6

# Slicing rows and columns
row_slice = arr[1]     # Output: [4 5 6]
col_slice = arr[:, 1]  # Output: [2 5 8]

# Boolean indexing
bool_index = arr[arr > 5]  # Output: [6 7 8 9]

#Array Shape Manipulation
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Reshaping an array
reshaped_arr = arr.reshape(3, 2)

# Transposing an array
transposed_arr = arr.T

# Flattening an array
flattened_arr = arr.flatten()

#Linear Algebra Operations
arr = np.array([[1, 2], [3, 4]])

# Matrix determinant
determinant = np.linalg.det(arr)

# Matrix inverse
inverse = np.linalg.inv(arr)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(arr)

# Solving linear equations
A = np.array([[2, 3], [1, -1]])
b = np.array([8, 1])
solution = np.linalg.solve(A, b)


# Importing Data

# Load data from a text file
data = np.loadtxt('data.txt', delimiter=',')
# Load data from a CSV file
data = np.genfromtxt('data.csv', delimiter=',')
# Load data from a NumPy binary file
data = np.load('data.npy')

# Exporting Data

# Exporting data to a text file
# Assuming 'data' is your NumPy array
np.savetxt('data.txt', data, delimiter=',')
# Exporting data to a CSV file
# Assuming 'data' is your NumPy array
np.savetxt('data.csv', data, delimiter=',')
# Exporting data to a NumPy Binary file
# Assuming 'data' is your NumPy array
np.save('data.npy', data)
