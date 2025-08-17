import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import torch
from torch import nn
import joblib
from model_train_pytorch import GainPredictor

class QuadcopterDynamics:
    def __init__(self):
        self.m = 2.0      # Kütle (kg)
        self.g = 9.8      # Yerçekimi ivmesi (m/s^2)
        self.l = 0.2      # Rotorlar arası mesafenin yarısı (m)
        self.I1 = 1.25    # Pitch ekseni atalet momenti (N s^2/rad)
        self.I2 = 1.25    # Roll ekseni atalet momenti (N s^2/rad)
        self.I3 = 2.50    # Yaw ekseni atalet momenti (N s^2/rad)
        self.k1, self.k2, self.k3 = 0.010, 0.010, 0.010 # Öteleme sürüklenme katsayıları
        self.k4, self.k5, self.k6 = 0.012, 0.012, 0.012 # Dönme sürüklenme katsayıları
        
        self.steady_state_u = np.array([self.g, 0, 0, 0])

    def state_derivative(self, state, control_inputs):
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
        
        vx_dot = u1 * (c_psi * s_th * c_phi + s_psi * s_phi) - (self.k1 / self.m) * vx
        vy_dot = u1 * (s_psi * s_th * c_phi - c_psi * s_phi) - (self.k2 / self.m) * vy
        vz_dot = u1 * (c_th * c_phi) - self.g - (self.k3 / self.m) * vz
        
        omega_theta_dot = u2 - (self.k4 * self.l / self.I1) * omega_theta
        omega_phi_dot   = u3 - (self.k5 * self.l / self.I2) * omega_phi
        omega_psi_dot   = u4 - (self.k6 / self.I3) * omega_psi

        return np.array([x_dot, y_dot, z_dot, theta_dot, phi_dot, psi_dot,
                         vx_dot, vy_dot, vz_dot, omega_theta_dot, omega_phi_dot, omega_psi_dot])

    def update_state(self, state, control_inputs, dt):
        state_derivatives = self.state_derivative(state, control_inputs)
        return state + state_derivatives * dt

import numpy as np

class BaseTrajectory:
    def get_reference(self, t):
        raise NotImplementedError

class CircularTrajectory(BaseTrajectory):
    def __init__(self, radius=5, altitude=10, speed=2.5, course_lock=False):
        self.radius = radius
        self.altitude = altitude
        self.speed = speed
        self.omega = self.speed / self.radius if self.radius != 0 else 0
        self.course_lock = course_lock
        self.total_lap_time = (2 * np.pi * self.radius) / self.speed

    def get_reference(self, t):
        ref_state = np.zeros(12)
        ref_x = self.radius * np.cos(self.omega * t)
        ref_y = self.radius * np.sin(self.omega * t)
        ref_z = self.altitude
        ref_vx = -self.radius * self.omega * np.sin(self.omega * t)
        ref_vy = self.radius * self.omega * np.cos(self.omega * t)

        ref_psi = 0.0
        if self.course_lock:
            ref_psi = np.arctan2(ref_vy, ref_vx)

        ref_state[[0, 1, 2, 5, 6, 7]] = [ref_x, ref_y, ref_z, ref_psi, ref_vx, ref_vy]
        return ref_state

class SquareTrajectory(BaseTrajectory):
    def __init__(self, side_length=10, altitude=10, speed=2.0, course_lock=False):
        self.side = side_length
        self.alt = altitude
        self.speed = speed
        self.time_per_side = self.side / self.speed if self.speed != 0 else float('inf')
        self.total_lap_time = 4 * self.time_per_side
        self.course_lock = course_lock

    def get_reference(self, t):
        ref_state = np.zeros(12)
        hs = self.side / 2.0
        time_in_lap = t % self.total_lap_time
        side_index = int(time_in_lap // self.time_per_side)
        time_on_side = time_in_lap % self.time_per_side
        
        if side_index == 0:
            ref_x, ref_y, ref_vx, ref_vy = -hs + self.speed * time_on_side, -hs, self.speed, 0
        elif side_index == 1:
            ref_x, ref_y, ref_vx, ref_vy = hs, -hs + self.speed * time_on_side, 0, self.speed
        elif side_index == 2:
            ref_x, ref_y, ref_vx, ref_vy = hs - self.speed * time_on_side, hs, -self.speed, 0
        else: # side_index == 3
            ref_x, ref_y, ref_vx, ref_vy = -hs, hs - self.speed * time_on_side, 0, -self.speed

        ref_psi = 0.0
        if self.course_lock:
            if not (ref_vx == 0 and ref_vy == 0):
                ref_psi = np.arctan2(ref_vy, ref_vx)

        ref_state[[0, 1, 2, 5, 6, 7]] = [ref_x, ref_y, self.alt, ref_psi, ref_vx, ref_vy]
        return ref_state

class BaseController:
    def __init__(self, quad_dynamics):
        self.quad_dynamics = quad_dynamics
        self.steady_state_u = quad_dynamics.steady_state_u

    def calculate_control(self, current_state, reference_state, dt):
        raise NotImplementedError

class LQRController(BaseController):
    def __init__(self, quad_dynamics, Q, R, psi_tol=np.pi/36):
        super().__init__(quad_dynamics)
        self.Q = Q
        self.R = R
        self.psi_tol = psi_tol
        self.psi_ss_current = 0.0
        self.K = self._solve_lqr_gain(self.psi_ss_current)

    def _create_A_matrix(self, psi_ss):
        A = np.zeros((12, 12))
        c_psi = np.cos(psi_ss)
        s_psi = np.sin(psi_ss)
        A[0:6, 6:12] = np.identity(6)
        A[6, 3] = self.quad_dynamics.g * c_psi
        A[6, 4] = self.quad_dynamics.g * s_psi
        A[7, 3] = self.quad_dynamics.g * s_psi
        A[7, 4] = -self.quad_dynamics.g * c_psi
        A[6, 6] = -self.quad_dynamics.k1 / self.quad_dynamics.m
        A[7, 7] = -self.quad_dynamics.k2 / self.quad_dynamics.m
        A[8, 8] = -self.quad_dynamics.k3 / self.quad_dynamics.m
        A[9, 9] = -self.quad_dynamics.k4 * self.quad_dynamics.l / self.quad_dynamics.I1
        A[10, 10] = -self.quad_dynamics.k5 * self.quad_dynamics.l / self.quad_dynamics.I2
        A[11, 11] = -self.quad_dynamics.k6 / self.quad_dynamics.I3
        return A

    def _create_B_matrix(self):
        B = np.zeros((12, 4))
        B[8, 0] = 1.0   # u1 -> vz_dot (aslında u1*(c_th*c_phi) -> vz_dot)
        B[9, 1] = 1.0   # u2 -> omega_theta_dot
        B[10, 2] = 1.0  # u3 -> omega_phi_dot
        B[11, 3] = 1.0  # u4 -> omega_psi_dot
        return B

    def _solve_lqr_gain(self, psi_ss):
        A = self._create_A_matrix(psi_ss)
        B = self._create_B_matrix()
        P = solve_continuous_are(A, B, self.Q, self.R)
        return np.linalg.inv(self.R) @ B.T @ P

    def calculate_control(self, current_state, reference_state, dt):
        if abs(current_state[5] - self.psi_ss_current) > self.psi_tol:
            self.psi_ss_current = current_state[5]
            self.K = self._solve_lqr_gain(self.psi_ss_current)
        
        x_delta = current_state - reference_state
        u_delta = -self.K @ x_delta
        return u_delta + self.steady_state_u
    

class PyTorchController(BaseController):
    def __init__(self, quad_dynamics, model_filename='lqr_pytorch_model.pth', scaler_X="scaler_X.pkl", scaler_y="scaler_y.pkl"):
        super().__init__(quad_dynamics)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch '{self.device}' cihazını kullanıyor.")

        # Modeli ve Ölçekleyicileri Yükle
        self.model = GainPredictor().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_filename, map_location=self.device))
            self.scaler_X = joblib.load(scaler_X)
            self.scaler_y = joblib.load(scaler_y)
        except FileNotFoundError as e:
            print(f"\nHATA: Gerekli bir dosya bulunamadı ({e.filename})!")
            raise
        
        self.model.eval()

    def calculate_control(self, current_state, reference_state, dt):
        current_psi = current_state[5]
        normalized_psi = np.arctan2(np.sin(current_psi), np.cos(current_psi))
        
        input_data = np.array([[normalized_psi]])
        scaled_input = self.scaler_X.transform(input_data)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            scaled_k_flat_tensor = self.model(input_tensor)
        
        scaled_k_flat = scaled_k_flat_tensor.cpu().numpy()
        rescaled_k_flat = self.scaler_y.inverse_transform(scaled_k_flat)
        
        K = rescaled_k_flat.reshape(4, 12)
        x_delta = current_state - reference_state
        u_delta = -K @ x_delta
        
        return u_delta + self.steady_state_u

class PIDController(BaseController):
    def __init__(self, quad_dynamics):
        super().__init__(quad_dynamics)
        self.Kp_z, self.Ki_z, self.Kd_z = 3.0, 0.5, 4.0
        self.Kp_att, self.Ki_att, self.Kd_att = 4.0, 0.6, 5.0
        self.Kp_yaw, self.Ki_yaw, self.Kd_yaw = 2.5, 0.2, 3.5
        self.Kp_pos = 0.05

        self.integral_error = np.zeros(4)
        self.previous_error = np.zeros(4)

    def calculate_control(self, current_state, reference_state, dt):
        z, theta, phi, psi = current_state[2:6]
        ref_z, ref_theta, ref_phi, ref_psi = reference_state[2:6]
        
        x_error = reference_state[0] - current_state[0]
        y_error = reference_state[1] - current_state[1]

        c_psi, s_psi = np.cos(psi), np.sin(psi)
        ref_theta_from_pos = self.Kp_pos * (x_error * c_psi + y_error * s_psi)
        ref_phi_from_pos   = self.Kp_pos * (x_error * s_psi - y_error * c_psi)

        error = np.array([ref_z - z, ref_theta_from_pos - theta, ref_phi_from_pos - phi, ref_psi - psi])
        
        self.integral_error += error * dt
        derivative_error = (error - self.previous_error) / dt
        self.previous_error = error

        u1_delta = self.Kp_z * error[0] + self.Ki_z * self.integral_error[0] + self.Kd_z * derivative_error[0]
        u2 = self.Kp_att * error[1] + self.Ki_att * self.integral_error[1] + self.Kd_att * derivative_error[1]
        u3 = self.Kp_att * error[2] + self.Ki_att * self.integral_error[2] + self.Kd_att * derivative_error[2]
        u4 = self.Kp_yaw * error[3] + self.Ki_yaw * self.integral_error[3] + self.Kd_yaw * derivative_error[3]
        
        u1_total = self.steady_state_u[0] + u1_delta
        return np.array([u1_total, u2, u3, u4])
    
class InterpolatedLQRController(BaseController):
    def __init__(self, quad_dynamics, dataset_filename='lqr_gains_dataset.csv'):
        super().__init__(quad_dynamics)
        import pandas as pd
        try:
            df = pd.read_csv(dataset_filename)
            self.psi_values = df['psi_ss'].values
            
            gain_columns = [col for col in df.columns if col.startswith('k_')]
            self.gains = df[gain_columns].values # Shape: (num_points, 48)

        except FileNotFoundError:
            print(f"HATA: Veri seti dosyası '{dataset_filename}' bulunamadı!")
            raise

    def calculate_control(self, current_state, reference_state, dt):
        current_psi = current_state[5]

        # Açıların -pi ve +pi arasında sürekli olması için normalizasyon yapıyoruz.
        # Bu, enterpolasyonun sınırlarda doğru çalışmasını sağlar.
        current_psi = np.arctan2(np.sin(current_psi), np.cos(current_psi))

        # 48 kazanç değerinin her birini anlık psi değerine göre enterpole ediyoruz.
        interpolated_k_flat = np.zeros(self.gains.shape[1])
        for i in range(self.gains.shape[1]):
            interpolated_k_flat[i] = np.interp(current_psi, self.psi_values, self.gains[:, i])

        K = interpolated_k_flat.reshape(4, 12)

        x_delta = current_state - reference_state
        u_delta = -K @ x_delta
        
        return u_delta + self.steady_state_u


class Simulation:
    def __init__(self, quad, controller, trajectory, total_time, dt):
        self.quad = quad
        self.controller = controller
        self.trajectory = trajectory
        self.total_time = total_time
        self.dt = dt
        self.num_steps = int(total_time / dt)

    def run(self, initial_state):
        current_state = initial_state.copy()
        history = [current_state]
        control_history = []
        
        u_steady_state = self.quad.steady_state_u 

        for i in range(self.num_steps):
            t = i * self.dt
            reference_state = self.trajectory.get_reference(t)
            
            control_inputs = self.controller.calculate_control(current_state, reference_state, self.dt)
            control_history.append(control_inputs)

            current_state = self.quad.update_state(current_state, control_inputs, self.dt)
            history.append(current_state.copy())
            
        return np.array(history), np.array(control_history)


if __name__ == '__main__':
    DT = 0.01
    TOTAL_TIME = 25.0 # Simülasyonun toplam süresi (saniye)
    TRAJECTORY_SPEED = 2 # Yörüngenin hızı (m/s)
    
    initial_state = np.zeros(12)

    quad_model = QuadcopterDynamics()

    trajectory_to_follow = CircularTrajectory(radius=5, altitude=10, speed=TRAJECTORY_SPEED, course_lock=False)
    # trajectory_to_follow = SquareTrajectory(side_length=10, altitude=10, speed=TRAJECTORY_SPEED, course_lock=True)
    
    # LQR Kontrolcü
    Q = np.diag([10, 10, 20, 1, 1, 5, 1, 1, 1, 1, 1, 1])
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    
    # controller_to_use = LQRController(quad_model, Q, R)
    controller_to_use = InterpolatedLQRController(quad_model, 'lqr_gains_dataset.csv')
    # controller_to_use = PIDController(quad_model)
    # controller_to_use = PyTorchController(quad_model, model_filename='best_model.pth', scaler_X="scaler_X.pkl", scaler_y="scaler_y.pkl")

    simulation = Simulation(quad=quad_model, 
                            controller=controller_to_use, 
                            trajectory=trajectory_to_follow,
                            total_time=TOTAL_TIME, 
                            dt=DT)
    
    simulation_history, control_history = simulation.run(initial_state)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ref_points = np.array([trajectory_to_follow.get_reference(t) for t in np.linspace(0, TOTAL_TIME, 200)])
    ax.plot(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], 'r--', label='Referans Yörünge')
    
    motor_spheres = [ax.plot([], [], [], 'o', markersize=5, color=c)[0] for c in ['b','g','c','m']]
    motor_arms_lines = [ax.plot([], [], [], 'k-')[0] for _ in range(4)]
    reference_sphere, = ax.plot([], [], [], 'o', markersize=8, color='red', alpha=0.7, label='Hedef Nokta')
    trajectory_line, = ax.plot([], [], [], 'b-', lw=2, label='Drone Yörüngesi')

    ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10]); ax.set_zlim([0, 15])
    ax.set_xlabel('X Ekseni (m)'); ax.set_ylabel('Y Ekseni (m)'); ax.set_zlabel('Z Ekseni (m)')
    ax.set_title(f'Kontrolcü: {type(controller_to_use).__name__} | Yörünge: {type(trajectory_to_follow).__name__}')
    ax.legend(); ax.grid(True)
    
    def rotation_matrix(phi, theta, psi):
        c_p, s_p = np.cos(phi), np.sin(phi); c_t, s_t = np.cos(theta), np.sin(theta); c_y, s_y = np.cos(psi), np.sin(psi)
        return np.array([[c_y*c_t, c_y*s_t*s_p - s_y*c_p, c_y*s_t*c_p + s_y*s_p],
                         [s_y*c_t, s_y*s_t*s_p + c_y*c_p, s_y*s_t*c_p - c_y*s_p],
                         [-s_t, c_t*s_p, c_t*c_p]])

    l = quad_model.l
    motor_positions_body = np.array([[l, 0, 0], [-l, 0, 0], [0, l, 0], [0, -l, 0]])

    def animate(i):
        step = max(1, int(len(simulation_history) / (TOTAL_TIME * 30)))
        i_sim = min(i * step, len(simulation_history) - 1)
        state = simulation_history[i_sim]
        x, y, z, theta, phi, psi = state[0:6]
        
        rot_mat = rotation_matrix(phi, theta, psi)
        motor_pos_world = (rot_mat @ motor_positions_body.T).T + np.array([x, y, z])
        for j in range(4):
            motor_spheres[j].set_data_3d([motor_pos_world[j, 0]], [motor_pos_world[j, 1]], [motor_pos_world[j, 2]])
            motor_arms_lines[j].set_data_3d([x, motor_pos_world[j,0]], [y, motor_pos_world[j,1]], [z, motor_pos_world[j,2]])
        
        ref_state = trajectory_to_follow.get_reference(i_sim * DT)
        reference_sphere.set_data_3d([ref_state[0]], [ref_state[1]], [ref_state[2]])
        trajectory_line.set_data_3d(simulation_history[:i_sim+1, 0], simulation_history[:i_sim+1, 1], simulation_history[:i_sim+1, 2])
        ax.view_init(elev=30, azim=i*0.2)
        return motor_spheres + motor_arms_lines + [reference_sphere, trajectory_line]

    frames_per_second = 30
    frames = int(TOTAL_TIME * frames_per_second)
    ani = FuncAnimation(fig, animate, frames=frames, interval=1000/frames_per_second, blit=False)
    
    
    # writer = FFMpegWriter(fps=frames_per_second, metadata=dict(artist='Alihan'), bitrate=1800)
    # ani.save('quad_simulation_final.mp4', writer=writer)

    plt.show()