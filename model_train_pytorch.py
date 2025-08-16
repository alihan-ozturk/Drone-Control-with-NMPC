import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os

# Adım 1: Veri Yükleme ve Hazırlama
try:
    df = pd.read_csv('lqr_gains_dataset.csv')
except FileNotFoundError:
    print("HATA: 'lqr_gains_dataset.csv' dosyası bulunamadı.")
    exit()

X = df[['psi_ss']].values
gain_columns = [col for col in df.columns if col.startswith('k_')]
y = df[gain_columns].values

# Adım 2: Veriyi Eğitim ve Test Setlerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Adım 3: Veri Normalizasyonu
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# Adım 4: PyTorch Tensor'larına ve DataLoader'lara Dönüştürme
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Adım 5: Yapay Sinir Ağı Modelinin Tanımlanması
class GainPredictor(nn.Module):
    def __init__(self, input_size=1, output_size=48):
        super(GainPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.network(x)

model = GainPredictor(input_size=X_train_tensor.shape[1], output_size=y_train_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_model_path = 'best_model.pth'

# Adım 6: Modelin Eğitilmesi ve En İyi Modelin Saklanması
print("Model eğitiliyor...")
epochs = 150
train_losses = []
test_losses = []
best_test_loss = float('inf')
best_epoch = -1
start_time = time.time()

for epoch in range(epochs):
    model.train()
    batch_train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_train_loss += loss.item()
    train_losses.append(batch_train_loss / len(train_loader))

    model.eval()
    batch_test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_test_loss += loss.item()
    
    current_test_loss = batch_test_loss / len(test_loader)
    test_losses.append(current_test_loss)
    
    # En iyi modeli kontrol et ve kaydet
    if current_test_loss < best_test_loss:
        best_test_loss = current_test_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Eğitim Hatası: {train_losses[-1]:.6f}, Test Hatası: {test_losses[-1]:.6f}')

end_time = time.time()
print(f"\nEğitim tamamlandı. Toplam süre: {end_time - start_time:.2f} saniye")
print(f"En iyi test hatası ({best_test_loss:.6f}) {best_epoch}. epoch'ta elde edildi.")

# Adım 7: En İyi Modelin Yüklenmesi ve Sonuçların Görselleştirilmesi
print(f"En iyi model '{best_model_path}' yükleniyor ve sonuçlar oluşturuluyor...")
model.load_state_dict(torch.load(best_model_path))

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Eğitim (Train) Loss')
plt.plot(test_losses, label='Test (Validation) Loss')
plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'En İyi Epoch ({best_epoch})')
plt.title('Eğitim ve Test Hatalarının Değişimi')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

model.eval()
with torch.no_grad():
    predictions_scaled = model(X_test_tensor)
    predictions = scaler_y.inverse_transform(predictions_scaled.numpy())
    y_test_original = scaler_y.inverse_transform(y_test_tensor.numpy())
    X_test_original = scaler_X.inverse_transform(X_test_tensor.numpy())

sort_indices = np.argsort(X_test_original.flatten())
num_gains = y_test_original.shape[1]
plots_per_figure = 16
num_figures = int(np.ceil(num_gains / plots_per_figure))

for fig_num in range(num_figures):
    plt.figure(figsize=(16, 12))
    start_idx = fig_num * plots_per_figure
    end_idx = min(start_idx + plots_per_figure, num_gains)
    plt.suptitle(f'En İyi Model ile Tahminler (Kazançlar {start_idx}-{end_idx-1})', fontsize=18)

    for i in range(start_idx, end_idx):
        subplot_idx = i % plots_per_figure + 1
        ax = plt.subplot(4, 4, subplot_idx)
        ax.plot(X_test_original[sort_indices], y_test_original[sort_indices, i], 'b-', label='Gerçek', linewidth=2)
        ax.plot(X_test_original[sort_indices], predictions[sort_indices, i], 'r--', label='Tahmin')
        ax.set_title(f'Kazanç: {gain_columns[i]}', fontsize=10)
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        if subplot_idx == 1:
            ax.set_xlabel('Psi Açısı (radyan)', fontsize=9)
            ax.set_ylabel('Kazanç Değeri', fontsize=9)
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

if os.path.exists(best_model_path):
    os.remove(best_model_path)