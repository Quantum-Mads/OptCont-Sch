"""
Hamiltonian simulation for finite-horizon LQR in error coordinates.

- Builds the Hamiltonian matrix C
- Computes lambda0 from the block exponential of e^{C T}
- Propagates [e(t); lambda(t)] using an ODE solver
- Recovers the optimal control u*(t) = -R^{-1} B^T lambda(t)

"""

import numpy as np
from numpy.linalg import solve
from scipy.linalg import expm
from scipy import linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# ------------------ Data ------------------
m = 1.0
M = 5.0
L = 2.0
g = -10.0
delta = 1.0

A = np.array([[0,    1,   0,  0],
              [0, -delta/M,  m*g/M,  0],                     # -0.2, -2.0
              [0,    0,   0,  1],
              [0, -delta/(M*L), -(M+m)*g/(M*L),  0]], float) # -0.1, +6.0
print(A)

B = np.array([[0.0],
              [1/M],          # 0.2
              [0.0],
              [1/(M*L)]], float)  # 0.1

Q = np.eye(4)
R = np.array([[1e-2]])  # 0.01
H = np.diag([0.0, 10, 50.0, 10.0])

x_ref = np.array([1.0, 0.0, np.pi, 0.0])
x0    = np.array([-1.0, 0.0, np.pi+0.1, 0.0])
e0    = x0 - x_ref  # initial error


T  = 10.0
dt = 0.001
N  = int(T/dt) + 1
t_eval = np.linspace(0.0, T, N)

# ------------------ Pre-calculations & Functions ------------------

RB  = la.solve(R, B.T)     # (1x4)  R^{-1}B^T
BRB = B @ RB          # (4x4)  B R^{-1}B^T
C  = np.block([[ A,    -BRB  ],
               [ -Q,   -A.T  ]])  # (8x8)


def Four_Partition(C, T):
    """
    C: matriz NxN
    T: Final time
    """
    N = C.shape[0] 
    s = N//2

    Phi_T = expm(C * T)    # e^{C T}= sum_{k=0}^{\infty} (C T)^k / k!
    U00 = Phi_T[:s, :s]    
    U01 = Phi_T[:s, s:]
    U10 = Phi_T[s:, :s]
    U11 = Phi_T[s:, s:]
    return Phi_T, U00, U01, U10, U11

def inverse_check(M):
    """ 
    Check if M is invertible by looking at its condition number.
    The condition number is the ratio of the largest to smallest singular value (Singular value: non-negative square root of eigenvalue of M^T M).
    If cond(M) is very large, M is close to singular and numerical errors may be amplified (M can strech or compress space too much in some directions).
    """
    cond_M = np.linalg.cond(M)
    if cond_M > 1e12:
        print(f"Warning: Matrix appears ill-conditioned (cond ~ {cond_M:.2e}).")
    return cond_M

def ODE(t, z):
    """"Combined dynamics (state and costate) for 8 dimensional linear ODE: z' = H @ z """
    return C @ z

def integrate_ODE(lam0, T, t_eval=None):
    z0 = np.concatenate((e0, lam0))
    sol = solve_ivp(
        ODE,                # the function defining z' = f(t, z) = C @ z
        (0.0, T),           # integrate from t = 0.0 to t = T
        z0,                 # initial condition z(0) = [e0, lambda0]
        t_eval=t_eval,      # times where to sample the solution (e.g. [T] -> only at final time)
        method='Radau',     # Radau (implicit) is stable/efficient for stiff linear Hamiltonian systems (although it is more computationally expensive than RK45)
        jac=lambda t, z: C, # constant Jacobian = H; speeds up Newton iterations and improves robustness
        rtol=1e-12,          # relative tolerance (tight for accurate terminal constraint)
        atol=1e-14          # absolute tolerance (tight to keep λ(T) accurate)
    )
    return sol

# ---------- Compute lambda0 from block exponential at T ----------
Phi_T, U00, U01, U10, U11 = Four_Partition(C, T)

M = (U11 - H @ U01)
N = (H @ U00 - U10)
inverse_check(M)

lambda0 = la.solve(M, N @ e0)   # lambda0 = M^{-1} N e0
# should be: [-5.54745912; -5.83952384; 44.49282174; 16.83104340]


# ---------- Integrate combined ODE for z = [e; lambda] ----------
sol = integrate_ODE(lambda0, T, t_eval=t_eval)
z_T = sol.y[:, -1]


E   = sol.y[:4, :].T         # state errors
LAM = sol.y[4:, :].T         # costates
X   = E + x_ref              # states
U   = -(LAM @ RB.T).ravel()  # optimal control u = -R^{-1} B^T lambda


# References
lam_T = LAM[-1]   # Final costate (should be close to H*e(T))
e_T   = E[-1]     # Final error (should be small)
X_T  = X[-1]     # Final state (should be close to x_ref)

HeT   = H @ E[-1]



# ------------------ Graph 1: e(t), lambda(t), x(t) ------------------
fig, axs = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

# 1) e(t) vs 0
for i in range(4):
    axs[0].plot(t_eval, E[:, i], label=f"e{i+1}(t)")
axs[0].axhline(0, color='k', linestyle='--', label="0")
axs[0].set_ylabel("e(t)")
axs[0].set_title("State errors e(t) vs 0")
axs[0].legend(); axs[0].grid(True)

# 2) lambda(t) vs H*e(T)
for i in range(4):
    axs[1].plot(t_eval, LAM[:, i], label=f"λ{i+1}(t)")
    axs[1].axhline(HeT[i], linestyle='--', label=f"(H e(T))_{i+1}")
axs[1].set_ylabel("λ(t)")
axs[1].set_title("Costados λ(t) vs H·e(T)")
axs[1].legend(); axs[1].grid(True)

# 3) x(t) vs x_ref
for i in range(4):
    axs[2].plot(t_eval, X[:, i], label=f"x{i+1}(t)")
    axs[2].axhline(x_ref[i], linestyle='--', label=f"x_ref[{i+1}]")
axs[2].set_xlabel("t [s]"); axs[2].set_ylabel("x(t)")
axs[2].set_title("States x(t) vs x_ref")
axs[2].legend(); axs[2].grid(True)

plt.tight_layout()
plt.show()



# ------------------ Graph 2: states x(t), control input u(t) ------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column

# States
ax1.plot(sol.t, X[:,0], label='x')
ax1.plot(sol.t, X[:,1], label='v')
ax1.plot(sol.t, X[:,2], label=r'$\theta$')
# theta_wrapped = (X[:,2] + np.pi) % (2*np.pi) - np.pi
# ax1.plot(sol.t, theta_wrapped, label=r'$\theta$ wrapped')
ax1.plot(sol.t, X[:,3], label=r'$\omega$')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('State')
ax1.set_title("States x(t)")
ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
ax1.grid(True)

# Control input
ax2.plot(sol.t, U)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("u (control input)")
ax2.set_title("Control Input u(t) = -K e(t)")
plt.grid(True)

plt.tight_layout()
plt.show()


# # ---------- Print summary ----------
print("\n--- Summary ---")
print("lambda_0 =", lambda0)
print("z(T) =", z_T)
print("x(T) =", X_T)
print("x_ref =", x_ref)

print("\nCond(U11 - H U01) =", np.linalg.cond(M))

print("\n‖e(T)‖ = ‖x(T) - x_ref‖ =", np.linalg.norm(X_T - x_ref))
print("‖λ(T) - H e(T)‖ =", np.linalg.norm(lam_T - HeT))


