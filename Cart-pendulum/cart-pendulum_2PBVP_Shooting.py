import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve
from scipy import linalg as la


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

B = np.array([[0.0],
              [1/M],          # 0.2
              [0.0],
              [1/(M*L)]], float)  # 0.1

Q = np.eye(4)
R = np.array([[1e-2]])  # 0.01

x_ref = np.array([1.0, 0.0, np.pi, 0.0])
x0    = np.array([-1.0, 0.0, np.pi + 0.1, 0.0])
e0    = x0 - x_ref  # initial error

T  = 10.0
dt = 0.001
N  = int(T/dt) + 1
t_eval = np.linspace(0.0, T, N)

# ------------------ Pre-calculations & Functions ------------------

RB  = la.solve(R, B.T)     # (1x4)  R^{-1}B^T
BRB = B @ RB          # (4x4)  B R^{-1}B^T
H  = np.block([[ A,    -BRB  ],
               [ -Q,   -A.T  ]])  # (8x8)

def u_of_lambda(lam): 
    """ Optimal control from PMP: u* = -R^{-1} B^T lam."""
    u = - RB @ lam
    # Alternatively, you can compute u directly using the formula:
    # u = -la.inv(R) @ B.T @ lam
    return u
def ODE(t, z):
    """"Combined dynamics (state and costate) for 8 dimensional linear ODE: z' = H @ z """
    return H @ z

# ------------------ Shooting function: we want lam(T) = 0 (residual ~ 0) ------------------
def integrate_with_lam0(lam0_guess, T, t_eval=None):
    z0 = np.concatenate((e0, lam0_guess))
    sol = integrate.solve_ivp(
        ODE,                # the function defining z' = f(t, z) = H @ z
        (0.0, T),           # integrate from t = 0.0 to t = T
        z0,                 # initial condition z(0) = [e0, lambda0_guess]
        t_eval=t_eval,      # times where to sample the solution (e.g. [T] -> only at final time)
        method='Radau',     # Radau (implicit) is stable/efficient for stiff linear Hamiltonian systems (although it is more computationally expensive than RK45)
        jac=lambda t, z: H, # constant Jacobian = H; speeds up Newton iterations and improves robustness
        rtol=1e-8,          # relative tolerance (tight for accurate terminal constraint)
        atol=1e-10          # absolute tolerance (tight to keep λ(T) accurate)
    )
    return sol

def shooting_error(lam0_guess):
    sol = integrate_with_lam0(lam0_guess,T, t_eval=[T])
    # sol.y is a (8,1) array with the final state and costate
    lam_T = sol.y[4:, -1]  # only costate at final time λ(T)
    return lam_T           # we want this to be zero


def newton_shooting(lam0_init, maxit=10, eps=1e-6):
    lam0 = lam0_init.copy()
    for k in range(maxit):
        r = shooting_error(lam0)  # 4x1
        nrm = la.norm(r)
        # print(f"Iter {k}: ‖r‖={nrm:.3e}")
        if nrm < 1e-10:
            break
        # Jacobiano numérico S ≈ dλ(T)/dλ(0) por diferencias finitas
        # (5 integraciones por iteración: base + 4 perturbaciones)
        S = np.zeros((4,4))
        h = 1e-6
        for j in range(4):
            d = np.zeros(4); d[j] = h
            r_j = shooting_error(lam0 + d)
            S[:, j] = (r_j - r) / h
        # Paso de Newton: lam0 <- lam0 - S^{-1} r
        try:
            delta = la.solve(S, r)
        except la.LinAlgError:
            # regularización mínima si S es mal condicionada
            delta = la.lstsq(S + 1e-12*np.eye(4), r)[0]
        lam0 = lam0 - delta
        if la.norm(delta) < eps*(1+la.norm(lam0)):
            break
    return lam0








# ===================== 1) Shooting method with random initial guess (sensitivity analysis) =====================
#rng = np.random.default_rng(0)  # seed for reproducibility
#lam0_bad = rng.uniform(-1, 1, size=4)
lam0_bad = [0, 0, 0, 0]  # Random initial guess for λ(0)
print("lambda(0):", lam0_bad)
r_bad = shooting_error(lam0_bad)
print("‖λ(T)‖ with a random guess:", np.linalg.norm(r_bad))

# Using fsolve to find a better initial guess
lam0_sol, info, integerflag, msg = fsolve(
    shooting_error,
    lam0_bad,
    full_output=True,
    maxfev=5000,    # increase maximum function evaluations
    xtol=1e-12,     # tighter tolerance on lambda(0)
    factor=0.1      # smaller initial step size (can help convergence)
)
if integerflag != 1: # Check if the solution converged
    print("\nFSOLVE warning:", msg)
print("Found lambda(0):", lam0_sol)
r_sol = shooting_error(lam0_sol)
print("‖λ(T)‖ with a fsolve guess:", np.linalg.norm(r_sol))


sol_init =integrate_with_lam0(lam0_sol, T, t_eval=t_eval)

E_init   = sol_init.y[:4, :].T       # (N,4) error states
LAM_init = sol_init.y[4:, :].T       # (N,4) costates
X_init = E_init + x_ref              # (N,4) actual states
U_init = -(LAM_init @ RB.T).ravel()  # U = -(R^{-1} B^T) λ  

# ===================== 2) Shooting method with almost exact initial guess =====================
# Note: This exact λ(0) formula using expm(H*T) only applies to constant linear time invariant systems with λ(T)=0 terminal condition.
# 
# For constant Hamiltonian H (LTI case):
# [ e(T) ]   = [ Φ11  Φ12 ]   [ e0 ]
# [ λ(T) ]     [ Φ21  Φ22 ]   [ λ0 ]
# 
# Imposing λ(T) = 0 gives:
# 0 = Φ21 * e0 + Φ22 * λ0
# ⇒ λ0 = -Φ22^{-1} * Φ21 * e0

Phi   = la.expm(H*T)                       # e^{H T}
Phi21 = Phi[4:, :4]
Phi22 = Phi[4:, 4:]
lam0_exact = -la.solve(Phi22, Phi21 @ e0)  # Assures λ(T)=0 in exact arithmetic
print("\nInitial guess for λ(0) (exact):", lam0_exact)
lam0_exact_pert = lam0_exact + 100*np.random.default_rng(0).standard_normal(4)
print("Initial guess for λ(0) (almost exact):", lam0_exact_pert)
# Comprobación directa en T (sin integrar):
z0  = np.concatenate([e0, lam0_exact])
zT  = Phi @ z0
lamT_direct = zT[4:]
print("‖λ(T)‖ (direct with expm):", np.linalg.norm(lamT_direct))

lam0_ns = newton_shooting(lam0_exact_pert, maxit=100)
r_ns = shooting_error(lam0_ns)
print("‖λ(T)‖ tras Newton-shooting:", np.linalg.norm(r_ns))

sol_ns  = integrate_with_lam0(lam0_ns,  T, t_eval=t_eval)
LAM_ns  = sol_ns.y[4:, :].T
U_ns    = -(LAM_ns  @ RB.T).ravel()



norm_init  = np.linalg.norm(LAM_init,  axis=1)
lamT_init   = norm_init[-1]
norm_ns  = np.linalg.norm(LAM_ns,  axis=1)
lamT_ns   = norm_ns[-1]


# ===================== 3) Graphs =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col')
# Left column axes
ax1L = axes[0, 0]  # costates (left)
ax2L = axes[1, 0]  # control  (left)

# Right column axes (exact copy)
ax1R = axes[0, 1]  # costates (right)
ax2R = axes[1, 1]  # control  (right)

# ---- Left column ----
for i in range(4):
    ax1L.plot(sol_init.t, LAM_init[:, i], label=rf'$\lambda_{{{i+1}}}(t)$')
ax1L.set_ylabel(r'$\lambda(t)$')
ax1L.set_title(fr'Costates λ(t) & Control u(t) Shooting without exp: ‖$\lambda(T)$‖={lamT_init:.1e}')
ax1L.grid(True)
ax1L.legend()

ax2L.plot(sol_init.t, U_init, color='k')
ax2L.set_xlabel('Tiempo [s]')
ax2L.set_ylabel(r'$u(t)$')
ax2L.grid(True)
# ---- Right column ----
for i in range(4):
    ax1R.plot(sol_ns.t, LAM_ns[:, i], label=rf'$\lambda_{{{i+1}}}(t)$')
ax1R.set_ylabel(r'$\lambda(t)$')
ax1R.set_title(fr'Costates λ(t) & Control u(t) Shooting with exp: ‖$\lambda(T)$‖={lamT_ns:.1e}')
ax1R.grid(True)
ax1R.legend()

ax2R.plot(sol_ns.t, U_ns, color='k')
ax2R.set_xlabel('Tiempo [s]')
ax2R.set_ylabel(r'$u(t)$')
ax2R.grid(True)

plt.tight_layout()
plt.show()


