import numpy as np
import pandas as pd  # Pandas kütüphanesini import ediyoruz
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from main2 import QuadcopterDynamics, LQRController, CircularTrajectory, Simulation, InterpolatedLQRController

def run_multiple_simulations(min_speed, max_speed, num_simulations):
    speeds = np.linspace(min_speed, max_speed, num_simulations)
    simulation_results = []
    
    print(f"Simülasyonlar başlatılıyor. Hız aralığı: {min_speed:.2f} m/s - {max_speed:.2f} m/s. Toplam {num_simulations} simülasyon.")

    for i, speed in enumerate(speeds):
        quad_model = QuadcopterDynamics()

        #trajectory_to_follow = SquareTrajectory(side_length=10, altitude=10, speed=speed, course_lock=True)
        trajectory_to_follow = CircularTrajectory(radius=5, altitude=10, speed=speed, course_lock=True)

        total_time = trajectory_to_follow.total_lap_time
        if total_time == float('inf') or total_time < 0.1: # Çok kısa simülasyonları atla
            print(f"Hız {speed:.2f} m/s için geçersiz simülasyon süresi. Bu adım atlanıyor.")
            continue

        initial_state = trajectory_to_follow.get_reference(0)

        Q = np.diag([10, 10, 20, 1, 1, 5, 1, 1, 1, 1, 1, 1])
        R = np.diag([0.1, 0.1, 0.1, 0.1])
        
        # controller_to_use = LQRController(quad_model, Q, R)
        controller_to_use = InterpolatedLQRController(quad_model, 'lqr_gains_dataset.csv')
        # controller_to_use = PyTorchController(quad_model, model_filename='best_model.pth', scaler_X="scaler_X.pkl", scaler_y="scaler_y.pkl")
        
        simulation = Simulation(quad=quad_model, 
                                controller=controller_to_use, 
                                trajectory=trajectory_to_follow,
                                total_time=total_time, 
                                dt=0.01)
        
        history, control_history = simulation.run(initial_state)
        
        dt = 0.01
        num_steps_run = len(history)
        ref_time_vector = np.arange(0, num_steps_run * dt, dt)
        ref_points = np.array([trajectory_to_follow.get_reference(t) for t in ref_time_vector])
        
        x_delta = history - ref_points
        u_delta = control_history - quad_model.steady_state_u

        state_costs = np.einsum('ij,jk,ik->i', x_delta, Q, x_delta)
        control_costs = np.einsum('ij,jk,ik->i', u_delta, R, u_delta)

        mean_state_cost = np.mean(state_costs)
        mean_control_cost = np.mean(control_costs)
        total_mean_cost = mean_state_cost + mean_control_cost


        simulation_results.append({
            'speed': speed,
            'history': history,
            'reference': ref_points,
            'time_vector': np.arange(0, dt * len(history), dt),
            'mean_state_cost': mean_state_cost,
            'mean_control_cost': mean_control_cost,
            'total_mean_cost': total_mean_cost
        })
        print(f"Simülasyon {i+1}/{num_simulations} tamamlandı (Hız: {speed:.2f} m/s, Süre: {total_time:.2f} s).")
        
    return simulation_results

def display_cost_table(simulation_results):
    df = pd.DataFrame(simulation_results)
    
    # Sadece ilgili sütunları seç ve yeniden adlandır
    cost_df = df[['speed', 'mean_state_cost', 'mean_control_cost', 'total_mean_cost']].copy()
    cost_df.rename(columns={
        'speed': 'Hız (m/s)',
        'mean_state_cost': 'Ort. Durum Maliyeti (xᵀQx)',
        'mean_control_cost': 'Ort. Kontrol Maliyeti (uᵀRu)',
        'total_mean_cost': 'Toplam Ortalama Maliyet'
    }, inplace=True)

    cost_df.index = [f"Sim #{i+1}" for i in range(len(cost_df))]
    
    pd.options.display.float_format = '{:,.4f}'.format

    print("\n\n" + "="*80)
    print(" " * 20 + "SİMÜLASYON MALİYET ANALİZİ TABLOSU")
    print("="*80)
    print(cost_df)
    print("="*80 + "\n")


def plot_results(simulation_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Farklı Hızlarda Drone Yörünge Simülasyonu', fontsize=16)

    cmap = plt.get_cmap('plasma')
    speeds = [res['speed'] for res in simulation_results]
    min_speed, max_speed = min(speeds), max(speeds)
    
    last_ref = simulation_results[-1]['reference']
    x_min, x_max = last_ref[:, 0].min(), last_ref[:, 0].max()
    y_min, y_max = last_ref[:, 1].min(), last_ref[:, 1].max()
    margin_xy = 2.0
    
    ax1.set_title('X-Y Düzleminde Drone Yörüngeleri')
    ax1.set_xlabel('X Ekseni (m)')
    ax1.set_ylabel('Y Ekseni (m)')
    ax1.plot(last_ref[:, 0], last_ref[:, 1], 'k--', linewidth=1.5, label='Referans Yörünge')
    
    for result in simulation_results:
        history = result['history']
        speed = result['speed']
        normalized_speed = (speed - min_speed) / (max_speed - min_speed) if max_speed > min_speed else 0.5
        color = cmap(normalized_speed)
        ax1.plot(history[:, 0], history[:, 1], color=color)

    ax1.set_xlim(x_min - margin_xy, x_max + margin_xy)
    ax1.set_ylim(y_min - margin_xy, y_max + margin_xy)
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()

    ax2.set_title('Zamanla Yükseklik Değişimi (Z Ekseni)')
    ax2.set_xlabel('Zaman (s)')
    ax2.set_ylabel('Yükseklik (m)')

    z_alt = simulation_results[0]['reference'][0, 2]
    max_time = max(res['time_vector'][-1] for res in simulation_results if len(res['time_vector']) > 0)
    ax2.axhline(y=z_alt, color='k', linestyle='--', linewidth=1.5, label='Referans Yükseklik')

    all_z_values = []
    for result in simulation_results:
        history = result['history']
        time_vector = result['time_vector']
        normalized_speed = (result['speed'] - min_speed) / (max_speed - min_speed) if max_speed > min_speed else 0.5
        color = cmap(normalized_speed)
        ax2.plot(time_vector, history[:, 2], color=color)
        all_z_values.extend(history[:, 2])

    z_max_actual = np.max(all_z_values)
    z_margin = 0.2 
    ax2.set_ylim(0, z_max_actual + z_margin)
    ax2.set_xlim(0, max_time)
    ax2.grid(True)
    ax2.legend()

    fig.subplots_adjust(bottom=0.2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min_speed, vmax=max_speed))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Hız (m/s)', fontsize=12)

    plt.savefig("simulation_results_gs_circular.png")
    plt.show()

if __name__ == '__main__':
    MIN_HIZ = 1.0
    MAX_HIZ = 6.0
    SIMULASYON_SAYISI = 10

    results = run_multiple_simulations(MIN_HIZ, MAX_HIZ, SIMULASYON_SAYISI)
    
    if results:
        display_cost_table(results)
        plot_results(results)
    else:
        print("Simülasyon sonuçları üretilemedi, analiz yapılamıyor.")