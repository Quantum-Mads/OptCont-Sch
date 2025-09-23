import numpy as np
import math


def binary_form_of_integer(n, n_qubits):
    r"""
    The aux function to transform decimal number to binary

    IMPORTANT

    the function automatically add zeros before the number to fit the computation basis of n_qubits format
    """
    return bin(n)[2:].zfill(n_qubits)

def swap_dims(n, a, b):
    dims = list(range(n))
    dims[a], dims[b] = dims[b], dims[a]
    return tuple(dims)


def generate_matrix_p_int(n, m):
    r"""
    Generate the integer part of the operator matrix hat{p} with m-th power
    :param n: The number of qubits
    :param m: The matrix powers
    :return: \hat{p}^m regardless of the factor (-i/(2\delta x))^m
    """

    # The size of the matrix is 2^n x 2^n
    size = 2 ** n
    # Create a zero matrix of the specified size with a complex data type
    matrix = np.zeros((size, size), dtype=int)
    # Set the off-diagonal elements
    np.fill_diagonal(matrix[1:], -1)
    np.fill_diagonal(matrix[:, 1:], 1)
    # Set the corner elements to complete the specified structure
    matrix[0, -1] = -1
    matrix[-1, 0] = 1
    return np.linalg.matrix_power(matrix, m)

def generate_p_values(n, m, left_side_coordinate, right_side_coordinate):
    r"""
    This function generates the sparse indices of the matrix \hat{p} and its corresponding values
    :param n: The number of qubits
    :param m: The matrix power
    :param left_side_coordinate: The low bound of the domain
    :param right_side_coordinate: The upper bound of the domain
    :return: sparse_indices.tolist() : A list containing the sparse indices of the matrix \hat{p}
             p_values.tolist() : It's corresponding values
             l : The number of qubits for sparse indices
    """

    first_row = generate_matrix_p_int(n, m)[0]
    sparse_indices_temp = np.nonzero(first_row)[0]
    p_values_temp = first_row[first_row != 0]

    l = math.ceil(math.log2(len(sparse_indices_temp)))

    #If the sparsity is not an integer power of two we need to regard so zero elements as non-zero elements
    sparse_indices = sparse_indices_temp
    index = 0
    while len(sparse_indices) < 2 ** l:
        if not index in sparse_indices:
            sparse_indices = np.append(sparse_indices, index)
        index += 1
    sparse_indices.sort()
    p_values = np.zeros(2 ** l, dtype = int)
    for i in range(len(sparse_indices_temp)):
        p_values[np.where(sparse_indices == sparse_indices_temp[i])[0][0]] = p_values_temp[i]

    a = left_side_coordinate  # left side coordinate
    b = right_side_coordinate  # right side coordinate
    delta_z = (b - a) / 2 ** n  # grid size

    p_values = p_values * (-1j / (2 * delta_z)) ** m

    return sparse_indices.tolist(), p_values.tolist(), l

def generate_p_values_for_multiple_m(n, m, left_side_coordinate, right_side_coordinate):
    r"""
        This function generates the sparse indices of the matrix \hat{p} and its corresponding values for multiple m values
        :param n: The number of qubits
        :param m: The matrix powers, it's a list
        :param left_side_coordinate: The low bound of the domain
        :param right_side_coordinate: The upper bound of the domain
        :return: spare_indices_multi : A 2D list containing the sparse indices of the matrix hat{p}
                 pm_values_multi : It's corresponding values
                 l_multi : The number of qubits for sparse indices
        """
    sparse_indices_multi = []
    pm_values_multi = []
    l_multi = []

    for i in range(len(m)):
        spare_indices_temp, pm_values_temp, l_temp = generate_p_values(n, m[i], left_side_coordinate,
                                                                       right_side_coordinate)
        sparse_indices_multi += [spare_indices_temp]
        pm_values_multi += [pm_values_temp]
        l_multi += [l_temp]
    return sparse_indices_multi, pm_values_multi, l_multi

def generate_sparse_indices_for_hamiltonian(n, m):
    r"""
    This function generates the sparse indices of the whole hamiltonian
    :param n: The number of qubits
    :param m: The matrix powers, it's a list
    :return:
    """
    sparse_indices = []
    for order in m:
        sparse_indices += np.nonzero(generate_matrix_p_int(n, order)[0])[0].tolist()
    sparse_indices = sorted(set(sparse_indices))
    l = math.ceil(math.log2(len(sparse_indices)))
    # If the sparsity is not an integer power of two we need to regard so zero elements as non-zero elements
    index = 0
    while len(sparse_indices) < 2 ** l:
        if not index in sparse_indices:
            sparse_indices += [index]
        index += 1
    sparse_indices.sort()

    return sparse_indices, l


# generate the coordinate matrix \hat{x} with normalization term
def generate_diagonal_matrix(left, right, n, q):

    a = left  # left side coordinate
    b = right - (right - left) / 2 ** n  # right side coordinate, because left_coordinate = x0 < x1 < ... < xN < x_{N+1} = right_coordinate, where N = 2^n

    delta_z = (b - a) / (2 ** n - 1)  # grid size
    # Generate a sequence. Note that b+z is used because np.arange does not include the endpoint
    sequence = np.arange(a, b + delta_z * 5 / 6, delta_z, dtype=np.complex128)

    # Here N_psi is the normalization constant. If we set m=2^n-1 then N_psi^2=(m+1)a^2+a\deta_z*m(m+1)+\delta_z^2*m(m+1)(2m+1)/6
    N_psi = 2 ** n * a ** 2 + a * delta_z * (2 ** n - 1) * (2 ** n) + delta_z ** 2 * (2 ** n - 1) * (2 ** n) * (2 ** (
            n + 1) - 1) / 6
    N_psi = np.sqrt(N_psi)

    sequence = sequence/N_psi

    # Create a diagonal matrix from the sequence
    return np.linalg.matrix_power(np.diag(sequence), q)

def generate_matrix_p(n, m, a, b):
    r"""
    Generate the integer part of the operator matrix hat{p} with m-th power
    :param n: The number of qubits
    :param m: The matrix powers
    :return: \hat{p}^m regardless of the factor (-i/(2\delta x))^m
    """

    # The size of the matrix is 2^n x 2^n
    size = 2 ** n
    # Create a zero matrix of the specified size with a complex data type
    matrix = np.zeros((size, size), dtype=np.complex128)
    # Set the off-diagonal elements
    np.fill_diagonal(matrix[1:], -1)
    np.fill_diagonal(matrix[:, 1:], 1)
    # Set the corner elements to complete the specified structure
    matrix[0, -1] = -1
    matrix[-1, 0] = 1

    delta_z = (b - a) / 2 ** n  # grid size
    matrix = matrix * (-1j/(2 * delta_z))
    return np.linalg.matrix_power(matrix, m)