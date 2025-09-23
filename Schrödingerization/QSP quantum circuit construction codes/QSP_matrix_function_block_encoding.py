import numpy as np
from QSP_Phase_factor_Solver import QSP_solver
from fundamental_gates_functions import dagger
from numpy.polynomial.chebyshev import poly2cheb


def QSP_block_encoding(is_coef_cheby, coef, parity, opts, u, n, m):
    """
        This function use QSP to construct the block-encoding of matrix functions of Hermitian matrix.
        Parameters:
        - is_coef_cheby (bool): True: The polynomial coefficients is given under the Chebyshev polynomials of the first kind basis (T_0, T_1, ...);
                                False: The polynomial coefficients is given under normal basis (1, x, x^2, x^3, ...).
        - coef : Coefficients of the polynomial (Chebyshev basis or normal basis), it should be in ascending order, contains all coefficients (even and odd parts).
        - parity : Parity of the polynomial (0 -- even, 1 -- odd)
        - opts: Dictionary containing options:
            - criteria: Stopping criteria
            - useReal: Whether to use only real number operations
            - targetPre: Whether Pre is the target function
            - method: Choose 'LBFGS', 'FPI', or 'Newton'
            - typePhi: Full or reduced phase
        - u : Initial block-encoding of the Hermitian matrix
        - n : The number of qubits in the Hermitian matrix being block-encoded
        - m : The number of the auxiliary qubits in the block-encoding
        :return: phase modulation gate sequence
        """
    # Remove odd or even order terms based on parity
    if parity == 0:
        coef[1::2] = [0] * (len(coef[1::2]))  # Even polynomial: remove odd-order terms
    else:
        coef[::2] = [0] * (len(coef[::2]))  # Odd polynomial: remove even-order terms
    if is_coef_cheby:
        coef_cheby = coef
    else:
        coef_cheby = poly2cheb(coef)  # Find chebyshev coefficients for the given polynomial
    # Retrieve coefficients based on parity
    coef_cheby = coef_cheby[parity::2]  # Extract coefficients of either even or odd terms

    phi_proc, out = QSP_solver(coef_cheby, parity, opts) #This phi_proc is the symmetric phase factors in arXiv:2002.11649, Page 8, Eq. 24.
    d = len(phi_proc) - 1
    phi = [0] * d
    phi[0] = phi_proc[0] + phi_proc[-1] + (d - 1) * np.pi / 2
    for i in range(1, d):
        phi[i] = phi_proc[i] - np.pi / 2   # This is exactly the phase factors we want, see arXiv:1806.01838v1, Page 12, Cor. 8 for more details.

    U_phi = u_phi(phi, parity, u, m)
    U_minus_phi = u_phi([-phi_i for phi_i in phi], parity, u, m)
    gate_sequence = [{'name': 'H', 'target': [0], 'control': [], "control_sequence": []}]
    gate_sequence += [{'name': 'zoom_in', "block_gate_sequence": U_phi, 'target': list(range(1, n + m + 1)),
                       'control': [0], 'control_sequence': [0]}]
    gate_sequence += [{'name': 'zoom_in', "block_gate_sequence": U_minus_phi, 'target': list(range(1, n + m + 1)),
                       'control': [0], 'control_sequence': [1]}]
    gate_sequence += [{'name': 'H', 'target': [0], 'control': [], "control_sequence": []}]

    return gate_sequence

def gate_z_pi(m, theta):
    gate_sequence = [{"name": 'X', 'target': [0], 'control':[], "control_sequence":[]}]
    gate_sequence += [{"name": 'phase_gate', "parameter": 2 * theta, 'target': [0], 'control':list(range(1, m)), "control_sequence":[0] * (m - 1)}]
    gate_sequence += [{"name": 'X', 'target': [0], 'control': [], "control_sequence": []}]
    gate_sequence += [{"name": 'global_phase', "parameter": -theta, 'target': [0], 'control':[], "control_sequence":[]}]
    return gate_sequence


def u_phi(phi, parity, u, m):
    d = len(phi)
    U_phi = []
    if parity == 0:
        for i in range(int(d / 2), 0, -1):
            U_phi += u
            U_phi += gate_z_pi(m, phi[2 * i - 1])
            U_phi += dagger(u)
            U_phi += gate_z_pi(m, phi[2 * i - 2])
    else:
        for i in range(int((d - 1) / 2), 0, -1):
            U_phi += u
            U_phi += gate_z_pi(m, phi[2 * i])
            U_phi += dagger(u)
            U_phi += gate_z_pi(m, phi[2 * i - 1])
        U_phi += u
        U_phi += gate_z_pi(m, phi[0])
    return U_phi