import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

m = 2.0      # Quadcopter kütlesi (kg)
g = 9.8      # Yerçekimi ivmesi (m/s^2)
l = 0.2      # Bitişik rotorlar arası mesafenin yarısı (m)
I1 = 1.25    # Pitch (Yunuslama) ekseni etrafındaki atalet momenti (N s^2/rad)
I2 = 1.25    # Roll (Yuvarlanma) ekseni etrafındaki atalet momenti (N s^2/rad)
I3 = 2.50    # Yaw (Sapma) ekseni etrafındaki atalet momenti (N s^2/rad)
k1, k2, k3 = 0.010, 0.010, 0.010 # Öteleme sürüklenme katsayıları (N s/m)
k4, k5, k6 = 0.012, 0.012, 0.012 # Dönme sürüklenme katsayıları (N s/m)

# LQR Kontrolcü Ağırlık Matrisleri
Q = np.diag([
    10, 10, 20, # x, y, z pozisyon hatalarına verilen önem
    1, 1, 5,    # theta, phi, psi açısal pozisyon hatalarına verilen önem
    1, 1, 1,    # vx, vy, vz hız hatalarına verilen önem
    1, 1, 1     # Açısal hız hatalarına verilen önem
])
R = np.diag([0.1, 0.1, 0.1, 0.1]) # Kontrol girişlerinin (u1, u2, u3, u4) maliyetleri

dt = 0.01         # Zaman adımı (s)
total_time = 15.0 # Toplam simülasyon süresi (s)
psi_tol = np.pi / 6 # Kazanç güncellemesi için yaw açısı toleransı (radyan)

# [x, y, z, theta, phi, psi, vx, vy, vz, omega_theta, omega_phi, omega_psi]

def create_A_matrix(psi_ss):
    c_psi = np.cos(psi_ss)
    s_psi = np.sin(psi_ss)

    A = np.zeros((12, 12))
    # Üst-sağ blok: I_6 (Hızların pozisyonların türevi olduğunu belirtir)
    A[0:6, 6:12] = np.identity(6)

    # Alt-sol blok: Psi(ψ_1ss) matrisi
    A[6, 3] = g * c_psi
    A[6, 4] = g * s_psi
    A[7, 3] = g * s_psi
    A[7, 4] = -g * c_psi

    # Alt-sağ blok: Delta (Δ) matrisi (sürüklenme terimleri)
    A[6, 6] = -k1 / m
    A[7, 7] = -k2 / m
    A[8, 8] = -k3 / m
    A[9, 9] = -k4 * l / I1  # theta_dot (pitch hızı) sönümlemesi
    A[10, 10] = -k5 * l / I2 # phi_dot (roll hızı) sönümlemesi
    A[11, 11] = -k6 / I3     # psi_dot (yaw hızı) sönümlemesi
    
    return A

def create_B_matrix():
    B = np.zeros((12, 4))
    B[8, 0] = 1.0  # u1 -> vz_dot
    B[9, 1] = 1.0  # u2 -> omega_theta_dot (pitch ivmesi)
    B[10, 2] = 1.0 # u3 -> omega_phi_dot (roll ivmesi)
    B[11, 3] = 1.0 # u4 -> omega_psi_dot (yaw ivmesi)
    return B


def solve_lqr_gain(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

def quadcopter_state_equations(state, control_inputs):
    x, y, z, theta, phi, psi, vx, vy, vz, omega_theta, omega_phi, omega_psi = state
    u1, u2, u3, u4 = control_inputs

    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_th, s_th = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    x_dot = vx
    y_dot = vy
    z_dot = vz
    theta_dot = omega_theta
    phi_dot = omega_phi
    psi_dot = omega_psi
    
    vx_dot = u1 * (c_psi * s_th * c_phi + s_psi * s_phi) - (k1 / m) * vx
    vy_dot = u1 * (s_psi * s_th * c_phi - c_psi * s_phi) - (k2 / m) * vy
    vz_dot = u1 * (c_th * c_phi) - g - (k3 / m) * vz
    
    omega_theta_dot = u2 - (k4 * l / I1) * omega_theta
    omega_phi_dot   = u3 - (k5 * l / I2) * omega_phi
    omega_psi_dot   = u4 - (k6 / I3) * omega_psi

    return np.array([x_dot, y_dot, z_dot, theta_dot, phi_dot, psi_dot,
                       vx_dot, vy_dot, vz_dot, omega_theta_dot, omega_phi_dot, omega_psi_dot])

def run_simulation_with_trajectory_tracking(initial_state, Q, R, dt, total_time, psi_tol, radius=5, altitude=10):
    num_steps = int(total_time / dt)
    current_state = initial_state.copy()
    history = [current_state]
    steady_state_u = np.array([g, 0, 0, 0])
    psi_ss_current = initial_state[5]
    
    A_matrix = create_A_matrix(psi_ss_current)
    B_matrix = create_B_matrix()
    K = solve_lqr_gain(A_matrix, B_matrix, Q, R)

    for i in range(num_steps):
        if abs(current_state[5] - psi_ss_current) > psi_tol:
            psi_ss_current = current_state[5]
            A_matrix = create_A_matrix(psi_ss_current)
            K = solve_lqr_gain(A_matrix, B_matrix, Q, R)
            
        t = i * dt
        omega = 2 * np.pi / total_time
        ref_x, ref_y, ref_z = radius * np.cos(omega * t), radius * np.sin(omega * t), altitude
        ref_vx, ref_vy = -radius * omega * np.sin(omega * t), radius * omega * np.cos(omega * t)
        
        reference_state = np.zeros(12)
        reference_state[[0, 1, 2, 6, 7]] = [ref_x, ref_y, ref_z, ref_vx, ref_vy]
        
        x_delta = current_state - reference_state
        u_delta = -K @ x_delta
        control_inputs = u_delta + steady_state_u
        
        state_derivatives = quadcopter_state_equations(current_state, control_inputs)
        current_state += state_derivatives * dt
        history.append(current_state.copy())
        
    return np.array(history)

initial_state = np.zeros(12)
radius, altitude = 5, 10
simulation_history = run_simulation_with_trajectory_tracking(initial_state, Q, R, dt, total_time, psi_tol, radius, altitude)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
circle_theta_vis = np.linspace(0, 2 * np.pi, 200)
ax.plot(radius * np.cos(circle_theta_vis), radius * np.sin(circle_theta_vis), altitude, 'r--', label='reference')

motor_colors = ['blue', 'green', 'cyan', 'magenta']
motor_spheres = [ax.plot([], [], [], 'o', markersize=5, color=c)[0] for c in motor_colors]
motor_arms_lines = [ax.plot([], [], [], 'k-')[0] for _ in range(4)]
reference_sphere, = ax.plot([], [], [], 'o', markersize=8, color='red', alpha=0.7, label='Target')
trajectory_line, = ax.plot([], [], [], 'b-', lw=2, label='Drone Trajectory')

ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10]); ax.set_zlim([0, 15])
ax.set_xlabel('X Axis (m)'); ax.set_ylabel('Y Axis (m)'); ax.set_zlabel('Z Axis (m)')
ax.set_title('Trajectory Tracking'); ax.legend(); ax.grid(True)

def rotation_matrix(phi, theta, psi):
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    c_th, s_th = np.cos(theta), np.sin(theta)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    return np.array([
        [c_psi*c_th, c_psi*s_th*s_phi - s_psi*c_phi, c_psi*s_th*c_phi + s_psi*s_phi],
        [s_psi*c_th, s_psi*s_th*s_phi + c_psi*c_phi, s_psi*s_th*c_phi - c_psi*s_phi],
        [-s_th, c_th*s_phi, c_th*c_phi]
    ]) # [cite: 470]

motor_positions_body = np.array([[l, 0, 0], [-l, 0, 0], [0, l, 0], [0, -l, 0]])

def animate(i):
    step = max(1, int(len(simulation_history) / (total_time * 30)))
    i_sim = min(i * step, len(simulation_history) - 1)
    
    current_state = simulation_history[i_sim]
    x, y, z, theta, phi, psi = current_state[0:6]
    
    rot_mat = rotation_matrix(phi, theta, psi)
    motor_positions_world = (rot_mat @ motor_positions_body.T).T + np.array([x, y, z])
    for j in range(4):
        motor_spheres[j].set_data_3d([motor_positions_world[j, 0]], [motor_positions_world[j, 1]], [motor_positions_world[j, 2]])
        motor_arms_lines[j].set_data_3d([x, motor_positions_world[j,0]], [y, motor_positions_world[j,1]], [z, motor_positions_world[j,2]])
    
    t = i_sim * dt
    omega = 2 * np.pi / total_time
    ref_x, ref_y, ref_z = radius * np.cos(omega * t), radius * np.sin(omega * t), altitude
    reference_sphere.set_data_3d([ref_x], [ref_y], [ref_z])

    hist_x = simulation_history[:i_sim+1, 0]
    hist_y = simulation_history[:i_sim+1, 1]
    hist_z = simulation_history[:i_sim+1, 2]
    trajectory_line.set_data_3d(hist_x, hist_y, hist_z)
    
    ax.view_init(elev=30, azim=i*0.2)
    return motor_spheres + motor_arms_lines + [reference_sphere, trajectory_line]

frames_per_second = 30
frames = int(total_time * frames_per_second)
ani = FuncAnimation(fig, animate, frames=frames, interval=1000/frames_per_second, blit=False)

writer = FFMpegWriter(fps=frames_per_second, metadata=dict(artist='Alihan'), bitrate=1800)



ani.save('./quad_simulation_video.mp4', writer=writer)
plt.show()