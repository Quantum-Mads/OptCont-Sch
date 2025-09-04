import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg as la
from scipy import integrate
import matplotlib.pyplot as plt


# ------------------ Data ------------------ 

m = 1.0
M = 5.0
L = 2.0
g = -10.0     
delta = 1.0

A = np.array([[0,    1,   0,  0],
              [0, -delta/M,  m*g/M,  0],
              [0,    0,   0,  1],
              [0, -delta/(M*L),   -(M+m)*g/(M*L),  0]], dtype=float)

B = np.array([[0.0],
              [1/M],
              [0.0],
              [1/(M*L)]], dtype=float)

Q = np.eye(4)
R = np.array([[1e-2]]) 

x_ref = np.array([1.0, 0.0, np.pi, 0.0])
x0    = np.array([-1.0, 0.0, np.pi + 0.1, 0.0])


# ------------------ Stability & Controllability ------------------ 

# In a linear ODE (x' = A x) if any eigenvalue of A has a positive real part the system is unstable.
eigA = np.linalg.eigvals(A)  # Tells you if the system, without control, is stable or unstable.

# Builds the controllability matrix. This tells you whether you can fully control the system with the inputs you have.
Ctrb = np.hstack([B, A @ B, A @ (A @ B), A @ (A @ (A @ B))])
rankCtrb = np.linalg.matrix_rank(Ctrb)

print("\n=== Open-loop system properties ===")
print("Eigenvalues(A):", np.sort(eigA))
print("Controllability rank:", rankCtrb)


# ------------------ LQR ------------------ 

# 0 = A^T P + P A - P B R^-1 B^T P + Q
# u = -K x;  K = R^-1 B^T P
# x' = (A - B K)x

P = la.solve_continuous_are(A, B, Q, R)
print("ARE Solution: P =", P)

K = la.solve(R, B.T @ P).reshape(1,4)
print("\nLQR gain: K =", K)
# Alternatively, you can compute K directly using the formula:
#  K = la.inv(R) @ B.T @ P


# ------------------ Simulation ------------------ 

T  = 10.0
dt = 0.001
N  = int(T/dt) + 1
t_eval  = np.linspace(0.0, T, N)
t_span = (0.0, T)  # Time interval for the simulation

# If instead of x we have x - x_ref, then u = - K (x - x_ref) and x' = (A - B K) x + B K x_ref
# which is e' = (A - B K) e, where e = x - x_ref.

Acl = A - B @ K
def closed_loop_dynamics(t, e):  # e' = (A - B K) e
    return Acl @ e

e0 = x0 - x_ref  # Initial error

# Solve using RK45 (~ ode45)
sol = integrate.solve_ivp(closed_loop_dynamics, t_span, e0, method='RK45', t_eval=t_eval, vectorized=False)

# Signals
E = sol.y.T                      # error states
X = E + x_ref                    # actual states
U = - (E @ K.T)                  # control input u = -K e

# ------------------ Plots ------------------ 

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


