import numpy as np
import pandas as pd

from main2 import QuadcopterDynamics, LQRController

def generate_and_save_dataset(num_points=360, output_filename='lqr_gains_dataset.csv'):
    Q = np.diag([10, 10, 20, 1, 1, 5, 1, 1, 1, 1, 1, 1])
    R = np.diag([0.1, 0.1, 0.1, 0.1])

    quad_model = QuadcopterDynamics()
    lqr_calculator = LQRController(quad_model, Q, R)
    psi_values = np.linspace(-np.pi, np.pi, num_points)
    
    dataset = []

    for i, psi in enumerate(psi_values):
        K_matrix = lqr_calculator._solve_lqr_gain(psi)
        
        K_flat = K_matrix.flatten()
        
        row = np.concatenate(([psi], K_flat))
        dataset.append(row)
        
        if (i + 1) % (num_points // 10) == 0:
            print(f"{i + 1}/{num_points}")

    k_column_names = [f'k_{row}_{col}' for row in range(K_matrix.shape[0]) for col in range(K_matrix.shape[1])]
    column_names = ['psi_ss'] + k_column_names

    df = pd.DataFrame(dataset, columns=column_names)

    df.to_csv(output_filename, index=False)
    

    print(f"'{output_filename}' dosyasına kaydedildi.")
    print(f"Toplam {len(df)} satır veri üretildi.")

if __name__ == '__main__':
    generate_and_save_dataset(num_points=720, output_filename='lqr_gains_dataset.csv')