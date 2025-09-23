import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as cheb

def QSP_solver(coef, parity, opts):
    """
    Compute the QSP phase factor for the given polynomial.

    Parameters:
    - coef: Chebyshev polynomial coefficients in ascending order, it contains only even/odd coefficients.
    - parity: Parity of the polynomial (0 -- even, 1 -- odd)
    - opts: Dictionary containing options:
        - criteria: Stopping criteria
        - useReal: Whether to use only real number operations
        - targetPre: Whether Pre is the target function
        - method: Choose 'LBFGS', 'FPI', or 'Newton'
        - typePhi: Full or reduced phase

    Output:
    - phi_proc: Solution to the optimization problem, full phase factor if typePhi is True, or reduced phase factor if typePhi is False
    - out: Detailed solution information
    """

    # Set default options
    opts.setdefault('maxiter', 5e4)
    opts.setdefault('criteria', 1e-12)
    opts.setdefault('useReal', True)
    opts.setdefault('targetPre', True)
    opts.setdefault('method', 'FPI')
    opts.setdefault('typePhi', 'full')

    if opts['method'] == 'LBFGS':

        # Initial preparation
        tot_len = len(coef)
        delta = np.cos(np.arange(1, 2 * tot_len, 2) * (np.pi / (2 * 2 * tot_len)))

        if not opts['targetPre']:
            opts['target'] = lambda x, opts: -ChebyCoef2Func(x, coef, parity, True)
        else:
            opts['target'] = lambda x, opts: ChebyCoef2Func(x, coef, parity, True)

        opts['parity'] = parity

        obj = QSPObj_sym
        grad = QSPGrad_sym_real if opts['useReal'] else QSPGrad_sym

        # L-BFGS optimization
        start_time = time.time()
        phi, err, iter_count = QSP_LBFGS(obj, grad, delta, np.zeros(tot_len), opts)

        # Adjust the phase factor if it's an even function
        if parity == 0:
            phi[0] /= 2

        runtime = time.time() - start_time

    elif opts['method'] == 'FPI':
        phi, err, iter_count, runtime = QSP_CM(coef, parity, opts)

    elif opts['method'] == 'Newton':
        phi, err, iter_count, runtime = QSP_Newton(coef, parity, opts)

    else:
        print("The specified method does not exist. Please choose 'LBFGS', 'FPI', or 'Newton'.")
        return None, None

    # Output information
    out = {
        'iter': iter_count,
        'time': runtime,
        'value': err,
        'parity': parity,
        'targetPre': opts['targetPre']
    }

    # Return full or reduced phase factor
    if opts['typePhi'] == 'full':
        phi_proc = rdc_phase_factor_to_full(phi, parity, opts['targetPre'])
        out['typePhi'] = 'full'
    else:
        phi_proc = phi
        out['typePhi'] = 'reduced'

    return phi_proc, out


def ChebyCoef2Func(x, coef, parity, partialcoef):
    """
    Calculate the function value based on Chebyshev expansion coefficients.

    Input parameters:
    - x: Independent variable (array).
    - coef: Chebyshev coefficients.
    - parity: Parity of the polynomial (0 means even, 1 means odd).
    - partialcoef: Whether it contains only even/odd coefficients.

    Output:
    - ret: Function value at x.
    """
    if isinstance(x, np.ndarray):
        ret = np.zeros(len(x))
    else:
        ret = np.zeros(1)
    y = np.arccos(x)

    if partialcoef:
        if parity == 0:
            for k in range(len(coef)):
                ret += coef[k] * np.cos(2 * (k) * y)
        else:
            for k in range(len(coef)):
                ret += coef[k] * np.cos((2 * k + 1) * y)
    else:
        if parity == 0:
            for k in range(0, len(coef), 2):
                ret += coef[k] * np.cos((k) * y)
        else:
            for k in range(1, len(coef), 2):
                ret += coef[k] * np.cos((k) * y)

    return ret


def QSPGetUnit_sym(phi, x, parity):
    """
    Calculate the QSP unitary matrix based on the given phase vector and point x.

    Input parameters:
    - phi: The last half of the phase factor (if parity == 1, phi is a simplified phase factor. If parity == 0, phi(1) differs by a factor of 2).
    - x: Point to evaluate, x belongs to [-1, 1].
    - parity: Parity of the phase factor (0 -- even, 1 -- odd).

    Output:
    - qspmat: QSP unitary matrix.

    This function constructs the Quantum Signal Processing (QSP) unitary matrix based on the given phase factor and x.
    """

    # Calculate Wx matrix
    Wx = np.array([[x, 1j * np.sqrt(1 - x ** 2)],
                   [1j * np.sqrt(1 - x ** 2), x]])

    # Phase gate
    gate = np.array([[np.exp(1j * np.pi / 4), 0],
                     [0, np.conj(np.exp(1j * np.pi / 4))]])

    # Calculate the complex exponent of the phase factor
    expphi = np.exp(1j * phi)

    # If it's an odd polynomial
    if parity == 1:
        ret = np.array([[expphi[0], 0], [0, np.conj(expphi[0])]])
        for k in range(1, len(expphi)):
            ret = np.dot(ret, np.dot(Wx, np.array([[expphi[k], 0], [0, np.conj(expphi[k])]])))
        ret = np.dot(ret, gate)
        qspmat = np.dot(np.transpose(ret), np.dot(Wx, ret))

    # Even polynomial
    else:
        ret = np.eye(2)
        for k in range(1, len(expphi)):
            ret = np.dot(ret, Wx * np.array([expphi[k], np.conj(expphi[k])]))
        ret = np.dot(ret, gate)
        qspmat = np.dot(np.transpose(ret), np.dot(np.array([[expphi[0], 0], [0, np.conj(expphi[0])]]), ret))

    return qspmat


def QSPObj_sym(phi, delta, opts):
    """
    Calculate the QSP objective function value given the phase factor and sample points.

    Input parameters:
    - phi: Phase factor (if parity == 1, it is a simplified phase factor; if parity == 0, phi(1) differs by a factor of 2).
    - delta: Array of sample points.
    - opts: Dictionary structure containing the following fields:
        - target: Target function.
        - parity: Parity of the phase factor (0 -- even, 1 -- odd).

    Output:
    - obj: Objective function value.

    The objective function is computed by calculating the squared error at each sample point with respect to the target function value, returning the value at each point.
    """

    # Initialize the objective function value
    if isinstance(delta, np.ndarray):
        m = len(delta)
    else:
        m = 1
    obj = np.zeros(m)

    # Calculate the objective function value at each sample point
    for i in range(m):
        # Get the QSP unitary matrix
        qspmat = QSPGetUnit_sym(phi, delta[i], opts['parity'])
        # Compute the squared error between the (1,1) element and the target value
        obj[i] = 0.5 * (np.real(qspmat[0, 0]) - opts['target'](delta[i], opts)).item() ** 2

    return obj


def QSPGetPimDeri_sym_real(phi, x, parity):
    """
    Compute the imaginary part of the (1,1) element of the QSP unitary matrix and its Jacobian at the given point x.

    Input parameters:
    - phi: Simplified phase factor.
    - x: Evaluation point (x ∈ [-1, 1]).
    - parity: Parity (0 -- even, 1 -- odd).

    Output:
    - y: Values of Pim and its Jacobian at point x.
    """

    if isinstance(phi, np.ndarray):
        n = len(phi)
    else:
        n = 1
    theta = np.arccos(x)

    B = np.array([[np.cos(2 * theta), 0, -np.sin(2 * theta)],
                  [0, 1, 0],
                  [np.sin(2 * theta), 0, np.cos(2 * theta)]])

    L = np.zeros((n, 3))
    L[n - 1, :] = [0, 1, 0]

    for k in range(n - 2, -1, -1):
        L[k, :] = np.dot(L[k + 1, :], np.dot(
            [[np.cos(2 * phi[k + 1]), -np.sin(2 * phi[k + 1]), 0],
             [np.sin(2 * phi[k + 1]), np.cos(2 * phi[k + 1]), 0],
             [0, 0, 1]], B))

    R = np.zeros((3, n))

    if parity == 0:
        R[:, 0] = [1, 0, 0]
    else:
        R[:, 0] = [np.cos(theta), 0, np.sin(theta)]

    for k in range(1, n):
        R[:, k] = np.dot(B, np.dot(
            [[np.cos(2 * phi[k - 1]), -np.sin(2 * phi[k - 1]), 0],
             [np.sin(2 * phi[k - 1]), np.cos(2 * phi[k - 1]), 0],
             [0, 0, 1]], R[:, k - 1]))

    y = np.zeros(n + 1)

    for k in range(n):
        y[k] = 2 * np.dot(L[k, :], np.dot(
            [[-np.sin(2 * phi[k]), -np.cos(2 * phi[k]), 0],
             [np.cos(2 * phi[k]), -np.sin(2 * phi[k]), 0],
             [0, 0, 0]], R[:, k]))

    y[n] = np.dot(L[n - 1, :], np.dot(
        [[np.cos(2 * phi[n - 1]), -np.sin(2 * phi[n - 1]), 0],
         [np.sin(2 * phi[n - 1]), np.cos(2 * phi[n - 1]), 0],
         [0, 0, 1]], R[:, n - 1]))

    return y


def QSPGrad_sym_real(phi, delta, opts):
    """
    Compute the gradient and objective value of the QSP function for symmetric phase factors.

    Input parameters:
    - phi: Phase factor variable.
    - delta: Sample points.
    - opts: Dictionary containing the target function and the parity of the phase factor.

    Output:
    - grad: Gradient of the objective function.
    - obj: Objective function value.
    """

    if isinstance(delta, np.ndarray):
        m = len(delta)
    else:
        m = 1

    if isinstance(phi, np.ndarray):
        d = len(phi)
    else:
        d = 1

    obj = np.zeros(m)
    grad = np.zeros((m, d))
    targetx = opts['target']
    parity = opts['parity']

    # Treat the first phase factor in simplified form
    if parity == 0:
        phi[0] = phi[0] / 2

    # Compute gradient and objective value
    for i in range(m):
        x = delta[i]
        y = QSPGetPimDeri_sym_real(phi, x, parity)  # Compute Jacobian
        if parity == 0:
            y[0] = y[0] / 2
        y = -y  # Flip sign
        gap = y[-1] - targetx(x, opts)
        obj[i] = 0.5 * gap.item() ** 2
        grad[i, :] = y[:-1] * gap

    return grad, obj


def QSPGrad_sym(phi, delta, opts):
    """
    Compute the gradient and objective value of the QSP function for symmetric phase factors.

    Input parameters:
    - phi: Phase factor variable.
    - delta: Sample points.
    - opts: Dictionary containing the target function and the parity of the phase factor.

    Output:
    - grad: Gradient of the objective function.
    - obj: Objective function value.
    """

    if isinstance(delta, np.ndarray):
        m = len(delta)
    else:
        m = 1

    if isinstance(phi, np.ndarray):
        d = len(phi)
    else:
        d = 1

    obj = np.zeros(m)
    grad = np.zeros((m, d))
    gate = np.array([[np.exp(1j * np.pi / 4), 0], [0, np.conj(np.exp(1j * np.pi / 4))]])
    exptheta = np.exp(1j * phi)
    targetx = opts['target']
    parity = opts['parity']

    # Begin gradient calculation
    for i in range(m):
        x = delta[i]
        Wx = np.array([[x, 1j * np.sqrt(1 - x ** 2)], [1j * np.sqrt(1 - x ** 2), x]])

        tmp_save1 = np.zeros((2, 2, d), dtype=complex)
        tmp_save2 = np.zeros((2, 2, d), dtype=complex)

        tmp_save1[:, :, 0] = np.eye(2)
        tmp_save2[:, :, 0] = np.array([[exptheta[d - 1], 0], [0, np.conj(exptheta[d - 1])]]) @ gate

        for j in range(1, d):
            tmp_save1[:, :, j] = tmp_save1[:, :, j - 1] @ np.array(
                [[exptheta[j - 1], 0], [0, np.conj(exptheta[j - 1])]]) @ Wx
            tmp_save2[:, :, j] = np.array(
                [[exptheta[d - j - 1], 0], [0, np.conj(exptheta[d - j - 1])]]) @ Wx @ tmp_save2[:, :, j - 1]

        if parity == 1:
            qspmat = np.transpose(tmp_save2[:, :, d - 1]) @ Wx @ tmp_save2[:, :, d - 1]
            gap = np.real(qspmat[0, 0]) - targetx(x)
            leftmat = np.transpose(tmp_save2[:, :, d - 1]) @ Wx
            for j in range(d):
                grad_tmp = leftmat @ tmp_save1[:, :, j] @ np.array([[1j, 0], [0, -1j]]) @ tmp_save2[:, :, d - j - 1]
                grad[i, j] = 2 * np.real(grad_tmp[0, 0].item()) * gap
            obj[i] = 0.5 * gap ** 2
        else:
            qspmat = np.transpose(tmp_save2[:, :, d - 2]) @ Wx @ tmp_save2[:, :, d - 1]
            gap = np.real(qspmat[0, 0]) - targetx(x, opts)
            leftmat = np.transpose(tmp_save2[:, :, d - 2]) @ Wx
            for j in range(d):
                grad_tmp = leftmat @ tmp_save1[:, :, j] @ np.array([[1j, 0], [0, -1j]]) @ tmp_save2[:, :, d - j - 1]
                grad[i, j] = 2 * np.real(grad_tmp[0, 0].item()) * gap
            grad[i, 0] = grad[i, 0] / 2
            obj[i] = 0.5 * gap.item() ** 2

    return grad, obj


def QSP_LBFGS(obj, grad, delta, phi, opts):
    """
    Solve the phase factor optimization problem using the L-BFGS method.

    Input parameters:
    - obj: Objective function L(phi)
    - grad: Gradient of the objective function
    - delta: Sample points
    - phi: Initial values
    - opts: Dictionary containing options:
        - maxiter: Maximum number of iterations
        - gamma: Line search shrinkage rate
        - accrate: Line search acceptance rate
        - minstep: Minimum step size
        - criteria: Stopping criteria
        - lmem: L-BFGS memory size
        - print: Whether to print output
        - itprint: Print frequency
        - parity: Parity of the polynomial (0 -- even, 1 -- odd)
        - target: Target polynomial

    Output:
    - phi: Optimized phase factor
    - obj_value: Objective function value at the optimum
    - iter: Number of iterations
    """

    maxiter = opts.get('maxiter', 50000)
    gamma = opts.get('gamma', 0.5)
    accrate = opts.get('accrate', 1e-3)
    minstep = opts.get('minstep', 1e-5)
    crit = opts.get('criteria', 1e-12)
    lmem = opts.get('lmem', 200)
    pri = opts.get('print', 1)
    itprint = opts.get('itprint', 1)

    iter = 0
    d = len(phi)
    mem_size = 0
    mem_now = 0
    mem_grad = np.zeros((lmem, d))
    mem_obj = np.zeros((lmem, d))
    mem_dot = np.zeros(lmem)
    grad_s, obj_s = grad(phi, delta, opts)
    obj_value = np.mean(obj_s)
    GRAD = np.mean(grad_s, axis=0)

    if pri:
        print('L-BFGS solver started')

    while True:
        iter += 1
        theta_d = GRAD.copy()
        alpha = np.zeros(mem_size)

        for i in range(mem_size):
            subsc = (mem_now - i - 1) % lmem
            alpha[i] = mem_dot[subsc] * np.dot(mem_obj[subsc, :], theta_d)
            theta_d -= alpha[i] * mem_grad[subsc, :]

        theta_d *= 0.5

        if opts['parity'] == 0:
            theta_d[0] *= 2

        for i in range(mem_size):
            subsc = (mem_now - (mem_size - i - 1) - 1) % lmem
            beta = mem_dot[subsc] * np.dot(mem_grad[subsc, :], theta_d)
            theta_d += (alpha[mem_size - i - 1] - beta) * mem_obj[subsc, :]

        step = 1
        exp_des = np.dot(GRAD, theta_d)

        while True:
            theta_new = phi - step * theta_d
            obj_snew = obj(theta_new, delta, opts)
            obj_valuenew = np.mean(obj_snew)
            ad = obj_value - obj_valuenew
            if ad > exp_des * accrate * step or step < minstep:
                break
            step *= gamma

        phi = theta_new
        obj_value = obj_valuenew
        obj_max = np.max(obj_snew)
        grad_s, _ = grad(phi, delta, opts)
        GRAD_new = np.mean(grad_s, axis=0)
        mem_size = min(lmem, mem_size + 1)
        mem_now = mem_now % lmem + 1
        mem_grad[mem_now - 1, :] = GRAD_new - GRAD
        mem_obj[mem_now - 1, :] = -step * theta_d
        mem_dot[mem_now - 1] = 1 / np.dot(mem_grad[mem_now - 1, :], mem_obj[mem_now - 1, :])
        GRAD = GRAD_new

        if pri and iter % itprint == 0:
            if iter == 1 or (iter - itprint) % (itprint * 10) == 0:
                print(f"{'iter':>4} {'obj':>13} {'stepsize':>10} {'des_ratio':>10}")
            print(f"{iter:>4d}  {obj_max:+5.4e} {step:+3.2e} {ad / (exp_des * step):+3.2e}")

        if iter >= maxiter:
            print("Max iteration reached.")
            break
        if obj_max < crit ** 2:
            print("Stop criteria satisfied.")
            break

    return phi, obj_value, iter


def QSPGetPim_sym_real(phi, y, parity):
    """
    Get the imaginary part of the polynomial P in the QSP unitary matrix,
    based on the real matrix representation of the given simplified phase vector and point y in [-1, 1].
    y can be multiple points, in which case P will be a list.

    Parameters:
    - phi: Simplified phase factor
    - y: Input point(s)
    - parity: Parity (0 -- even, 1 -- odd)

    Returns:
    - Pim: Imaginary part of the (1,1) element of the QSP unitary matrix at y
    """
    Pim = np.zeros(len(y))
    n = len(phi)

    for m in range(len(y)):
        theta = np.arccos(y[m])
        B = np.array([
            [np.cos(2 * theta), 0, -np.sin(2 * theta)],
            [0, 1, 0],
            [np.sin(2 * theta), 0, np.cos(2 * theta)]
        ])

        if parity == 0:
            R = np.array([1, 0, 0])
        else:
            R = np.array([np.cos(theta), 0, np.sin(theta)])

        for k in range(1, n):
            R = np.dot(B, np.dot(np.array([
                [np.cos(2 * phi[k - 1]), -np.sin(2 * phi[k - 1]), 0],
                [np.sin(2 * phi[k - 1]), np.cos(2 * phi[k - 1]), 0],
                [0, 0, 1]
            ]), R))

        Pim[m] = np.dot([np.sin(2 * phi[n - 1]), np.cos(2 * phi[n - 1]), 0], R)

    return Pim


def QSPGetPim_sym(phi, y, parity):
    """
    Get the imaginary part of the polynomial P in the QSP unitary matrix,
    based on the given phase vector and point y in [-1, 1].
    y can be multiple points, in which case P will be a list.

    Parameters:
    - phi: Simplified phase factor
    - y: Input point(s)
    - parity: Parity (0 -- even, 1 -- odd)

    Returns:
    - Pim: Imaginary part of the (1,1) element of the QSP unitary matrix at y
    """
    Pim = np.zeros(len(y))
    phi = phi[::-1]
    expphi = np.exp(1j * phi)

    for n in range(len(y)):
        x = y[n]
        Wx = np.array([[x, 1j * np.sqrt(1 - x ** 2)],
                       [1j * np.sqrt(1 - x ** 2), x]])

        ret = np.array([expphi[0], 0])

        for k in range(1, len(expphi)):
            ret = np.dot(ret, np.dot(Wx, np.array([[expphi[k], 0], [0, np.conj(expphi[k])]])))

        if parity == 1:
            P = np.dot(np.dot(ret, Wx), ret.T)
        else:
            P = np.dot(ret, ret.T)

        Pim[n] = np.imag(P)

    return Pim


def F(phi, parity, opts):
    """
    Calculate the Chebyshev coefficients of P_im.
    P_im: The imaginary part of the (1,1) element of the QSP unitary matrix.

    Parameters:
    - phi: Simplified phase factor
    - parity: Parity (0 -- even, 1 -- odd)
    - opts: Dictionary containing:
        - useReal: If True, use QSPGetPim_sym_real, otherwise use QSPGetPim_sym

    Returns:
    - coe: Chebyshev coefficients corresponding to P_im
    """

    if 'useReal' not in opts:
        opts['useReal'] = True

    # Initial preparation
    d = len(phi)
    dd = 2 * d
    theta = np.arange(d + 1) * np.pi / dd
    M = np.zeros(2 * dd)

    # Select the calculation function
    if opts['useReal']:
        f = lambda x: QSPGetPim_sym_real(phi, x, parity)
    else:
        f = lambda x: QSPGetPim_sym(phi, x, parity)

    # Calculate the Chebyshev coefficients
    M[:d + 1] = f(np.cos(theta))
    M[d + 1:dd + 1] = (-1) ** parity * M[d - 1::-1]
    M[dd + 1:] = M[dd - 1:0:-1]

    M = np.fft.fft(M)  # FFT transform
    M = np.real(M)
    M = M / (2 * dd)
    M[1:-1] = M[1:-1] * 2
    coe = M[parity::2][:d]

    return coe


def QSP_CM(coef, parity, opts):
    """
    Solve the phase factor using contraction mapping, such that the real part of the (1,1) element of the QSP unitary matrix gives the desired Chebyshev expansion.

    Parameters:
    - coef: Chebyshev coefficients
    - parity: Parity (0 -- even, 1 -- odd)
    - opts: Dictionary containing:
        - maxiter: Maximum number of iterations
        - criteria: Stopping criteria
        - targetPre: Whether Pre is the target function
        - useReal: Whether to use real matrix multiplication to compute QSP terms
        - print: Whether to print output
        - itprint: Output frequency

    Returns:
    - phi: Simplified phase factor
    - err: Error (L1 norm)
    - iter: Number of iterations
    - runtime: Time used
    """

    maxiter = opts.get('maxiter', int(1e5))
    crit = opts.get('criteria', 1e-12)
    targetPre = opts.get('targetPre', True)
    useReal = opts.get('useReal', True)
    pri = opts.get('print', 1)
    itprint = opts.get('itprint', 1)

    start_time = time.time()

    # Initialization
    if targetPre:
        coef = -coef  # Flip the sign
    phi = coef / 2
    iter = 0

    # Iterative solution
    while True:
        Fval = F(phi, parity, opts)
        res = Fval - coef
        err = np.linalg.norm(res, 1)
        iter += 1

        if iter >= maxiter:
            print("Max iteration reached.")
            break
        if err < crit:
            print("Stop criteria satisfied.")
            break

        phi -= res / 2

        if pri and iter % itprint == 0:
            if iter == 1 or (iter - itprint) % (itprint * 10) == 0:
                print(f"{'iter':>4s}{'err':>13s}")
            print(f"{iter:>4d}  {err:+5.4e}")

    runtime = time.time() - start_time

    return phi, err, iter, runtime


def QSPGetPimDeri_sym(phi, x, parity):
    """
    Calculate Pim and its Jacobian value at point x
    P_im: Imaginary part of the (1,1) element of the QSP unitary matrix
    Note: theta must be a scalar value

    Parameters:
    - phi: Simplified phase factor
    - x: Input point
    - parity: Parity (0 -- even, 1 -- odd)

    Output:
    - y: Pim and its Jacobian values at point x
    """

    d = len(phi)
    expphi = np.exp(1j * phi)
    Wx = np.array([[x, 1j * np.sqrt(1 - x ** 2)],
                   [1j * np.sqrt(1 - x ** 2), x]])
    right = np.zeros((2, d), dtype=complex)
    y = np.zeros(d + 1, dtype=complex)
    right[:, -1] = np.array([expphi[-1], 0])

    for k in range(d - 2, -1, -1):
        right[:, k] = np.dot(np.dot(np.array([[expphi[k], 0], [0, np.conj(expphi[k])]]), Wx), right[:, k + 1])

    left = right[:, 0].T
    right = np.array([[1j, 0], [0, -1j]]) @ right

    if parity == 1:
        left = np.dot(left, Wx)

    for k in range(d - 1):
        y[k] = 2 * np.dot(left, right[:, k])
        left = np.dot(left @ np.array([[expphi[k], 0], [0, np.conj(expphi[k])]]), Wx)

    y[d - 1] = 2 * np.dot(left, right[:, d - 1])
    y[d] = np.dot(left, np.array([expphi[-1], 0]))

    y = np.imag(y)

    return y


def QSPGetPimDeri_sym_real(phi, x, parity):
    """
    Compute Pim and its Jacobian values at point x.
    P_im: Imaginary part of the (1,1) element of the QSP unitary matrix.
    Use the real matrix representation of Pim.
    Note: theta must be a scalar value.

    Parameters:
    - phi: Reduced phase factor
    - x: Input point
    - parity: Parity (0 -- even, 1 -- odd)

    Output:
    - y: Pim and its Jacobian values at point x
    """
    n = len(phi)
    theta = np.arccos(x)

    # Define the B matrix for the rotation operation
    B = np.array([
        [np.cos(2 * theta), 0, -np.sin(2 * theta)],
        [0, 1, 0],
        [np.sin(2 * theta), 0, np.cos(2 * theta)]
    ])

    # Initialize the L matrix, with the last row set to [0, 1, 0]
    L = np.zeros((n, 3))
    L[-1, :] = [0, 1, 0]

    # Compute the values of the L matrix, iterating from bottom to top
    for k in range(n - 2, -1, -1):
        L[k, :] = L[k + 1, :] @ np.array([
            [np.cos(2 * phi[k + 1]), -np.sin(2 * phi[k + 1]), 0],
            [np.sin(2 * phi[k + 1]), np.cos(2 * phi[k + 1]), 0],
            [0, 0, 1]
        ]) @ B

    # Initialize the R matrix, with the first column set based on parity
    R = np.zeros((3, n))
    if parity == 0:
        R[:, 0] = [1, 0, 0]
    else:
        R[:, 0] = [np.cos(theta), 0, np.sin(theta)]

    # Compute the values of the R matrix, iterating from left to right
    for k in range(1, n):
        R[:, k] = B @ (np.array([
            [np.cos(2 * phi[k - 1]), -np.sin(2 * phi[k - 1]), 0],
            [np.sin(2 * phi[k - 1]), np.cos(2 * phi[k - 1]), 0],
            [0, 0, 1]
        ]) @ R[:, k - 1])

    # Compute the values of y, including both the imaginary part and the Jacobian
    y = np.zeros(n + 1)
    for k in range(n):
        y[k] = 2 * L[k, :] @ np.array([
            [-np.sin(2 * phi[k]), -np.cos(2 * phi[k]), 0],
            [np.cos(2 * phi[k]), -np.sin(2 * phi[k]), 0],
            [0, 0, 0]
        ]) @ R[:, k]

    # Compute the final value of y
    y[-1] = L[-1, :] @ np.array([
        [np.cos(2 * phi[-1]), -np.sin(2 * phi[-1]), 0],
        [np.sin(2 * phi[-1]), np.cos(2 * phi[-1]), 0],
        [0, 0, 1]
    ]) @ R[:, -1]

    # Return the imaginary part of y
    return y


def F_Jacobian(phi, parity, opts):
    """
    Compute Pim and its Jacobian values at cos(theta).
    P_im: Imaginary part of the (1,1) element of the QSP unitary matrix.

    Parameters:
    - phi: Reduced phase factor
    - parity: Parity (0 -- even, 1 -- odd)
    - opts: Dictionary containing the 'useReal' field, deciding whether to use real number computation

    Output:
    - f: Value of F(phi)
    - df: Value of DF(phi)
    """
    # Set options, defaulting 'useReal' to True if not specified
    use_real = opts.get('useReal', True)

    # Decide which function to use based on options
    if use_real:
        f_func = lambda x: QSPGetPimDeri_sym_real(phi, x, parity)
    else:
        f_func = lambda x: QSPGetPimDeri_sym(phi, x, parity)

    d = len(phi)
    dd = 2 * d
    theta = np.arange(d + 1) * np.pi / dd  # Generate the theta vector
    M = np.zeros((2 * dd, d + 1))

    # Compute the first d+1 rows of the M matrix
    for n in range(d + 1):
        M[n, :] = f_func(np.cos(theta[n]))

    # Compute the remaining part of the M matrix
    M[d + 1: dd + 1, :] = ((-1) ** parity) * M[d - 1:: -1, :]
    M[dd + 1:, :] = M[dd - 1: 0: -1, :]

    # Perform FFT on M and extract the real part
    M = np.fft.fft(M, axis=0)
    M = np.real(M[: dd + 1, :])
    M[1:-1, :] *= 2
    M /= (2 * dd)

    # Extract the values of f and df
    f = M[parity: 2 * d: 2, -1]
    df = M[parity: 2 * d: 2, :-1]

    return f, df


def QSP_Newton(coef, parity, opts):
    """
    Newton's method to solve for the phase factor such that the real part of the (1,1) element of the QSP unitary matrix matches the desired Chebyshev expansion.

    Parameters:
    - coef: Chebyshev coefficients
    - parity: Parity (0 -- even, 1 -- odd)
    - opts: Dictionary containing the following fields:
      - maxiter: Maximum number of iterations
      - criteria: Stopping criteria
      - targetPre: Whether Pre is the target function
      - useReal: Use real matrix multiplication to get QSP terms
      - print: Whether to output
      - itprint: Print frequency

    Output:
    - phi: Reduced phase factor
    - err: Error (L1 norm)
    - iter: Number of iterations
    - runtime: Time used
    """
    # Set the options for the Newton solver
    maxiter = opts.get('maxiter', int(1e5))
    crit = opts.get('criteria', 1e-12)
    targetPre = opts.get('targetPre', True)
    useReal = opts.get('useReal', True)
    pri = opts.get('print', 1)
    itprint = opts.get('itprint', 1)

    start_time = time.time()

    # Initial preparation
    if targetPre:
        coef = -coef  # Reverse the target function

    phi = coef / 2
    iter = 0

    # Print format setup
    str_head = f"{'iter':>4s}{'err':>13s}\n"
    str_num = "{:>4d}  {:+5.4e} \n"

    # Solve using Newton's method
    while True:
        # Calculate the target function value Fval and its Jacobian DFval
        Fval, DFval = F_Jacobian(phi, parity, opts)
        res = Fval - coef
        err = np.linalg.norm(res, 1)  # Compute the error
        iter += 1

        # Check stopping conditions
        if iter >= maxiter:
            print("Max iteration reached.")
            break
        if err < crit:
            print("Stop criteria satisfied.")
            break

        # Update the phase factor
        phi = phi - np.linalg.solve(DFval, res)

        # Print current iteration information
        if pri and iter % itprint == 0:
            if iter == 1 or (iter - itprint) % (itprint * 10) == 0:
                print(str_head)
            print(str_num.format(iter, err))

    runtime = time.time() - start_time

    return phi, err, iter, runtime


def rdc_phase_factor_to_full(phi_cm, parity, targetPre):
    """
    Construct the full phase factor given the reduced phase factor.

    Parameters:
    - phi_cm: Reduced phase factor
    - parity: Parity (0 -- even, 1 -- odd)
    - targetPre: Whether Pre is the target function

    Returns:
    - phi_full: Full phase factor
    """

    phi_right = np.copy(phi_cm)

    if targetPre:
        phi_right[-1] += np.pi / 4

    dd = 2 * len(phi_right)

    if parity == 0:
        dd -= 1

    phi_full = np.zeros(dd)
    phi_full[-len(phi_right):] = phi_right
    phi_full[:len(phi_right)] += phi_right[::-1]

    return phi_full


def QSPGetUnitary(phase, x):
    """
    Get the real part of the (1,1) element of the QSP unitary matrix based on the given phase vector and x ∈ [-1, 1].

    Parameters:
    - phase: Phase vector
    - x: Input point x

    Output:
    - targ: Real part of the (1,1) element of the QSP unitary matrix (target approximation)
    """

    Wx = np.array([[x, 1j * np.sqrt(1 - x ** 2)],
                   [1j * np.sqrt(1 - x ** 2), x]])

    expphi = np.exp(1j * phase)

    ret = np.array([[expphi[0], 0], [0, np.conj(expphi[0])]])

    for k in range(1, len(expphi)):
        temp = np.array([[expphi[k], 0], [0, np.conj(expphi[k])]])
        ret = np.dot(np.dot(ret, Wx), temp)

    targ = np.real(ret[0, 0])

    return targ


def QSPGetEntry(xlist, phase, opts):
    """
    Get the approximate value of the (1,1) element of the QSP unitary matrix for the given phase vector and point list xlist.

    Parameters:
    - xlist: List of input points
    - phase: Full phase factor
    - opts: Dictionary containing phase factor information
        - targetPre: If True, get Pre; otherwise, get Pim
        - parity: Parity
        - typePhi: Phase type ('reduced' or 'full')

    Returns:
    - ret: Approximate target value of the QSP
    """

    typePhi = opts['typePhi']
    targetPre = opts['targetPre']
    parity = opts['parity']

    d = len(xlist)
    ret = np.zeros(d)

    if typePhi == 'reduced':
        dd = 2 * len(phase) - 1 + parity
        phi = np.zeros(dd)
        phi[-len(phase):] = phase
        phi[:len(phase)] += phase[::-1]
    else:
        phi = phase

    if not targetPre:
        phi[0] -= np.pi / 4
        phi[-1] -= np.pi / 4

    for i in range(d):
        x = xlist[i]
        ret[i] = QSPGetUnitary(phi, x)

    return ret