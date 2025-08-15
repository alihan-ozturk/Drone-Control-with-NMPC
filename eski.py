import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================
# System Parameters
# ======================
m = 1.0                # Mass (kg)
Ixx, Iyy, Izz = 0.1, 0.1, 0.1  # Moments of inertia (kg·m²)
g = 9.81               # Gravity (m/s²)
dt = 0.02              # Time step (s)
simulation_time = 20.0  # Simulation duration (s)
# (For a demo it is useful to shorten the simulation time.)

# ======================
# Circular Path Parameters
# ======================
r = 3.0                # Radius (m)
w = 1.0                # Angular velocity (rad/s)

# ======================
# Helper functions for rotations
# ======================
def rotation_matrix_np(phi, theta, psi):
    """Rotation matrix (world from body) using numpy for visualization."""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def rotation_matrix_casadi(phi, theta, psi):
    """Rotation matrix using CasADi symbolic expressions."""
    Rz = ca.vertcat( 
        ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
        ca.horzcat(ca.sin(psi),  ca.cos(psi), 0),
        ca.horzcat(0, 0, 1)
    )
    Ry = ca.vertcat( 
        ca.horzcat(ca.cos(theta), 0, ca.sin(theta)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(theta), 0, ca.cos(theta))
    )
    Rx = ca.vertcat( 
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
        ca.horzcat(0, ca.sin(phi),  ca.cos(phi))
    )
    return ca.mtimes(Rz, ca.mtimes(Ry, Rx))

# ======================
# Nonlinear Dynamics
# ======================
def nonlinear_dynamics(state, u):
    """
    Nonlinear quadrotor dynamics.
    state = [x, y, z, xd, yd, zd, phi, theta, psi, p, q, r]
    u = [T, tau_phi, tau_theta, tau_psi]
    """
    x, y, z, xd, yd, zd, phi, theta, psi, p, q, r_val = state
    T, tau_phi, tau_theta, tau_psi = u
    
    # Compute rotation matrix (body-to-world)
    R = rotation_matrix_np(phi, theta, psi)
    
    # Translational dynamics
    thrust_body = np.array([0, 0, T])
    thrust_world = R @ thrust_body
    acceleration = thrust_world / m - np.array([0, 0, g])
    
    # Rotational dynamics
    p_dot = (tau_phi - (Izz - Iyy) * q * r_val) / Ixx
    q_dot = (tau_theta - (Ixx - Izz) * p * r_val) / Iyy
    r_dot = (tau_psi - (Iyy - Ixx) * p * q) / Izz
    
    # Use simplified Euler angle derivatives: (phi_dot, theta_dot, psi_dot) = (p, q, r)
    return np.array([
        xd, yd, zd,
        acceleration[0], acceleration[1], acceleration[2],
        p, q, r_val,
        p_dot, q_dot, r_dot
    ])

# ======================
# CasADi Version of the Dynamics (for MPC)
# ======================
def quadrotor_dynamics(x, u):
    """
    CasADi symbolic dynamics.
    x: 12x1 state vector [x, y, z, xd, yd, zd, phi, theta, psi, p, q, r]
    u: 4x1 control vector [T, tau_phi, tau_theta, tau_psi]
    """
    # Extract state components
    pos = x[0:3]
    vel = x[3:6]
    angles = x[6:9]
    rates = x[9:12]
    phi = angles[0]
    theta = angles[1]
    psi = angles[2]
    
    T = u[0]
    tau_phi   = u[1]
    tau_theta = u[2]
    tau_psi   = u[3]
    
    # Compute rotation matrix symbolically
    R = rotation_matrix_casadi(phi, theta, psi)
    thrust_body = ca.vertcat(0, 0, T)
    thrust_world = ca.mtimes(R, thrust_body)
    acc = (1/m) * thrust_world - ca.vertcat(0, 0, g)
    
    # Euler angle derivatives (simplified)
    phi_dot   = rates[0]
    theta_dot = rates[1]
    psi_dot   = rates[2]
    
    # Rotational dynamics
    p = rates[0]
    q = rates[1]
    r_sym = rates[2]  # renamed to avoid conflict with symbol r
    p_dot = (tau_phi - (Izz - Iyy)*q*r_sym) / Ixx
    q_dot = (tau_theta - (Ixx - Izz)*p*r_sym) / Iyy
    r_dot = (tau_psi - (Iyy - Ixx)*p*q) / Izz
    
    x_dot = ca.vertcat(vel, acc, rates, ca.vertcat(p_dot, q_dot, r_dot))
    return x_dot

def f_discrete(x, u):
    """Discrete dynamics with simple Euler integration."""
    return x + dt * quadrotor_dynamics(x, u)

# ======================
# Reference Trajectory
# ======================
def get_reference(t):
    """
    Generate circular reference trajectory.
    Returns a 12-element vector representing the desired state.
    """
    x = r * np.cos(w * t)
    y = r * np.sin(w * t)
    z = 0.0
    xd = -r * w * np.sin(w * t)
    yd = r * w * np.cos(w * t)
    zd = 0.0
    xdd = -r * w**2 * np.cos(w * t)
    ydd = -r * w**2 * np.sin(w * t)
    # Desired roll and pitch (simplified)
    phi = ydd / g  
    theta = -xdd / g  
    psi = 0.0
    p = 0.0
    q = 0.0
    r_rate = 0.0
    return np.array([x, y, z, xd, yd, zd, phi, theta, psi, p, q, r_rate])


# Prediction horizon
N = 20

# Cost weights (same as used with LQR earlier)
Q = np.diag([10, 10, 10, 1, 1, 1, 5, 5, 1, 1, 1, 1])
R_weight = np.diag([0.1, 0.1, 0.1, 0.1])

# Create an optimization problem instance
opti = ca.Opti()

# Decision variables
X = opti.variable(12, N+1)  # state trajectory over horizon
U = opti.variable(4, N)     # control inputs over horizon

# Parameters (will be updated at each time step)
X0_param = opti.parameter(12)    # initial state
X_ref = opti.parameter(12, N+1)  # reference trajectory

# Build the cost function
cost = 0
for k in range(N):
    x_err = X[:, k] - X_ref[:, k]
    u_k = U[:, k]
    cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_k.T, R_weight, u_k])
# Terminal cost
x_err_terminal = X[:, N] - X_ref[:, N]
cost += ca.mtimes([x_err_terminal.T, Q, x_err_terminal])
opti.minimize(cost)

# Dynamics constraints (multiple shooting formulation)
for k in range(N):
    x_next = f_discrete(X[:, k], U[:, k])
    opti.subject_to(X[:, k+1] == x_next)

# Initial condition constraint
opti.subject_to(X[:, 0] == X0_param)

# (Optional) Input constraints:
T_min = 0
T_max = 2*m*g  # Allow thrust to be up to twice hover
tau_min = -1.0
tau_max = 1.0
for k in range(N):
    opti.subject_to(opti.bounded(T_min, U[0, k], T_max))
    opti.subject_to(opti.bounded(tau_min, U[1, k], tau_max))
    opti.subject_to(opti.bounded(tau_min, U[2, k], tau_max))
    opti.subject_to(opti.bounded(tau_min, U[3, k], tau_max))

# Set solver options for IPOPT
opts = {"ipopt.print_level": 0, "print_time": 0}
opti.solver('ipopt', opts)

def mpc_control(current_state, current_time):
    """
    Solve the MPC problem given the current state and time.
    Returns the first control input computed by the MPC.
    """
    # Build reference trajectory over the horizon
    ref_traj = np.zeros((12, N+1))
    for k in range(N+1):
        t_ref = current_time + k*dt
        ref_traj[:, k] = get_reference(t_ref)
    # Set parameter values
    opti.set_value(X0_param, current_state)
    opti.set_value(X_ref, ref_traj)
    
    # (Optional) Provide a simple initial guess
    opti.set_initial(X, ref_traj)
    # Use hover thrust as initial guess for all control moves
    hover = np.array([m*g, 0, 0, 0]).reshape(4,1)
    opti.set_initial(U, np.tile(hover, (1, N)))
    
    # Solve the optimization problem
    sol = opti.solve()
    u_opt = sol.value(U)
    return u_opt[:, 0]  # Return the first control input

# ======================
# Simulation Loop
# ======================
# Initial state: starting on the circular path at (r,0,0) with appropriate velocity.
state = np.array([r, 0, 0, 0, r * w, 0, 0, 0, 0, 0, 0, 0])
history = []
reference_history = []

# For each time step, solve the MPC, apply the first control input, and simulate the dynamics.
for t in np.arange(0, simulation_time, dt):
    x_ref = get_reference(t)
    u = mpc_control(state, t)
    
    # Simulate the continuous dynamics (Euler integration)
    derivative = nonlinear_dynamics(state, u)
    state = state + derivative * dt
    
    history.append(state.copy())
    reference_history.append(x_ref.copy())

history = np.array(history)
reference_history = np.array(reference_history)

# ======================
# Animation Setup (Visualization)
# ======================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_xlim(-r-1, r+1)
ax.set_ylim(-r-1, r+1)
ax.set_zlim(-1, 5)

# Plot the reference trajectory
ax.plot(reference_history[:, 0], reference_history[:, 1], reference_history[:, 2], 
        'r--', label='Reference Path')

# Elements for visualizing the quadrotor
arm_length = 0.5
arm1, = ax.plot([], [], [], 'k-', lw=2)
arm2, = ax.plot([], [], [], 'k-', lw=2)
ref_dot = ax.scatter([], [], [], c='red', s=100, label='Reference Point')
body = ax.scatter([], [], [], c='b', s=100)
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

# Colored markers to indicate orientation (front, back, right, left)
colors = ['red', 'green', 'blue', 'yellow']
markers = [ax.scatter([], [], [], c=color, s=50) for color in colors]

def update(frame):
    idx = frame
    state = history[idx]
    x, y, z, _, _, _, phi, theta, psi, _, _, _ = state
    R = rotation_matrix_np(phi, theta, psi)
    points_body = np.array([
        [arm_length, 0, 0],
        [-arm_length, 0, 0],
        [0, arm_length, 0],
        [0, -arm_length, 0]
    ])
    points_world = (R @ points_body.T).T + np.array([x, y, z])
    
    arm1.set_data([points_world[0, 0], points_world[1, 0]], 
                  [points_world[0, 1], points_world[1, 1]])
    arm1.set_3d_properties([points_world[0, 2], points_world[1, 2]])
    
    arm2.set_data([points_world[2, 0], points_world[3, 0]], 
                  [points_world[2, 1], points_world[3, 1]])
    arm2.set_3d_properties([points_world[2, 2], points_world[3, 2]])
    
    body._offsets3d = ([x], [y], [z])
    
    for i, marker in enumerate(markers):
        marker._offsets3d = ([points_world[i, 0]], [points_world[i, 1]], [points_world[i, 2]])
    
    # Update the red dot for the reference point
    ref_state = reference_history[idx]  # The reference state at the current frame
    ref_dot._offsets3d = ([ref_state[0]], [ref_state[1]], [ref_state[2]])
    
    time_text.set_text(f'Time: {idx*dt:.2f}s')
    # Return all updated artists. Note: add ref_dot to the returned tuple.
    return arm1, arm2, body, time_text, *markers, ref_dot


ani = FuncAnimation(fig, update, frames=len(history), interval=50, blit=False)
ani.save('quadrotor_simulation.mp4', writer='ffmpeg', fps=30)
plt.legend()
plt.show()

"""
# ======================
# Plot Error Magnitude Graphs
# ======================
errors = reference_history - history

# Calculate Euclidean Norm
position_error = np.linalg.norm(errors[:, 0:3], axis=1)
velocity_error = np.linalg.norm(errors[:, 3:6], axis=1)
angle_error = np.linalg.norm(errors[:, 6:9], axis=1)
angular_velocity_error = np.linalg.norm(errors[:, 9:12], axis=1)

time = np.arange(0, simulation_time, dt)

# Plotting the graphs
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig.suptitle('Euclidean Norms of State Errors (Over Time)', fontsize=16)

axs[0].plot(time, position_error, label='||Position Error||')
axs[0].set_ylabel('Position Error (m)')
axs[0].grid(True)

axs[1].plot(time, velocity_error, label='||Velocity Error||', color='orange')
axs[1].set_ylabel('Velocity Error (m/s)')
axs[1].grid(True)

axs[2].plot(time, angle_error, label='||Angle Error||', color='green')
axs[2].set_ylabel('Angle Error (rad)')
axs[2].grid(True)

axs[3].plot(time, angular_velocity_error, label='||Angular Velocity Error||', color='red')
axs[3].set_ylabel('Angular Velocity Error (rad/s)')
axs[3].set_xlabel('Time (s)')
axs[3].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
"""
