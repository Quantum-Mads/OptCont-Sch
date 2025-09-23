from QSP_matrix_function_block_encoding import QSP_block_encoding
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.special import jn


def qsp_hamiltonian_simulation(U_H, n, alpha, m, t, epsilon, beta, flag):
    """
    Suppose U_H is a (alpha, m, 0)-block-encoding of the time-independent Hamiltonian H, this function simulates e^{-iHt} or e^{iHt}
    up to precision 'epsilon', that is a (2/beta, m+2, epsilon)-block-encoding of the time-independent Hamiltonian H.
    :param U_H: A gat_sequence of the (alpha, m, 0)-block-encoding of the time-independent Hamiltonian H
    :param n: Number of register qubits
    :param alpha:
    :param m: Number of auxiliary qubits
    :param t: The time
    :param epsilon: The precision
    :param beta: To ensure numerical stability, we ought to make sure the block encoding function f(x) is strictly bounded by 1 over interval [-1, 1].
                 beta is acted as a precondition to guarantee beta*f(x) satisfies such limitation, in this function, beta ought to be in (0, 1).
    :param flag: flag == True : simulate e^{-iHt}; flag == False : simulate e^{iHt}
    :return:
    """
    if beta <= 0 or beta >= 1 :
        raise ValueError("beta must be between 0 and 1, strictly speaking (0, 1)!")
    t = alpha * t #exp(-iHt) = exp(-i(H/alpha) * (alpha * t))
    # Since in Hamiltonian simulation the function of interest is f(x) = exp(-ixt) = cos(xt) - i sin(xt),
    # we can implement cos(Ht) and sin(Ht) separately and then use LCU technique to combine them together


    #### Approximating the real component
    # Consider the real part beta * cos(t * x), whose L_infinity norm over [-1, 1] is strictly bounded by beta.
    parity = 0
    targ = lambda x: beta * np.cos(t * x)

    # Compute the Chebyshev coefficients up to d = 1.4 * |t| + log(1 / epsilon)
    # Such that the approximation error is bounded by epsilon_0
    d = int(np.ceil(1.4 * t + np.log(1/epsilon)))

    # Generate Chebyshev coefficients
    # x_vals = np.linspace(-1, 1, int(t*100))  # Sample x values between -1 and 1
    # f_vals = targ(x_vals)  # Evaluate the target function at sample points
    #
    # # Compute Chebyshev coefficients using least-squares fitting
    # # coef = np.polynomial.chebyshev.chebfit(x_vals, f_vals, d)
    # cheb = Chebyshev.fit(x_vals, f_vals, d)
    # coef = cheb.coef
    coef = np.zeros(d + 1)
    coef[0] = jn(0, t) * beta
    for i in range(1, d + 1):
        if i % 2 == 0:
            coef[i] = jn(i, t) * 2 * (-1) ** (i / 2) * beta


    opts = {
        'maxiter': 200,
        'criteria': 1e-13,
        'useReal': True,
        'targetPre': True,
        'method': 'FPI'
    }
    cos_Ht = QSP_block_encoding(True, coef, parity, opts, U_H, n, m)

    #### Approximating the imaginary component
    # Consider the imaginary part beta * sin(t * x), whose L_infinity norm over [-1, 1] is strictly bounded by beta.
    parity = 1
    targ = lambda x: beta * np.sin(t * x)

    # Compute the Chebyshev coefficients up to d = 1.4 * |t| + log(1 / epsilon)
    # Such that the approximation error is bounded by epsilon_0
    d = int(np.ceil(1.4 * t + np.log(1 / epsilon)))

    # Generate Chebyshev coefficients
    # x_vals = np.linspace(-1, 1, int(t*100))  # Sample x values between -1 and 1
    # f_vals = targ(x_vals)  # Evaluate the target function at sample points
    #
    # # Compute Chebyshev coefficients using least-squares fitting
    # # coef = np.polynomial.chebyshev.chebfit(x_vals, f_vals, d)
    # cheb = Chebyshev.fit(x_vals, f_vals, d)
    # coef = cheb.coef
    coef = np.zeros(d + 1)
    for i in range(d + 1):
        if i % 2 != 0:
            coef[i] = jn(i, t) * 2 * (-1) ** ((i - 1) / 2) * beta

    opts = {
        'maxiter': 200,
        'criteria': 1e-13,
        'useReal': True,
        'targetPre': True,
        'method': 'FPI'  # Fixed Point Iteration method
    }
    sin_Ht = QSP_block_encoding(True, coef, parity, opts, U_H, n, m)


    ####Use LCU technique to combine them together
    exp_iHt = [{'name': 'h', 'target': [0]}]
    exp_iHt += [{'name': 's', 'target': [0]}]
    if flag:
        exp_iHt += [{'name': 'z', 'target': [0]}]
    exp_iHt += [{'name': 'zoom_in', 'block_gate_sequence': cos_Ht, 'target': list(range(1, n + m + 2)),
                 'control': [0], 'control_sequence': [0]}]
    exp_iHt += [{'name': 'zoom_in', 'block_gate_sequence': sin_Ht, 'target': list(range(1, n + m + 2)),
                 'control': [0], 'control_sequence': [1]}]
    exp_iHt += [{'name': 'h', 'target': [0]}]

    factor = 2 / beta
    number_of_ancilla = m + 2
    number_of_qubits = n + m + 2

    return exp_iHt, factor, number_of_ancilla, number_of_qubits, d
