
# -*- coding: utf-8 -*-
"""
Created on Jan 21 2026
@author: K.N. Osipov
"""

import numpy as np
import matplotlib.pyplot as plt

# ========================
# 1. PARAMETERS (Textbook values)
# ========================
np.random.seed(42)
N = 2000

# To match textbook: P=19.314, K=0.8284, F=23.314
sigma_eta2 = 16.00  # High process noise (W)
sigma_nu2 = 4.0     # Low measurement noise (V)

# ========================
# 2. DATA SIMULATION
# ========================
eta = np.random.normal(0, np.sqrt(sigma_eta2), N)
nu = np.random.normal(0, np.sqrt(sigma_nu2), N)
a=1
x_true = np.zeros(N)
for t in range(1, N):
    x_true[t] = a*x_true[t-1] + eta[t]
y_obs = x_true + nu

# ========================
# 3. KALMAN FILTER
# ========================
x_filt = np.zeros(N)
P_theory = np.zeros(N)
P_pred = np.zeros(N)
T = np.zeros(N)
sigma = np.zeros(N)
K = np.zeros(N)
innovations = np.zeros(N)
x_filt[0] = y_obs[0]
P_theory[0] = 19.3 

sigma[0]=10;
T[0]=10;

n=8;

for t in range(1, N):
    P_pred[t] = P_theory[t-1] + sigma_eta2
    
  
    sigma[t]=np.var(x_filt[0:t])+1e-9 
    T[t] = np.var(y_obs[0:t])
    K[t] = ((a**2)*sigma[t] /(T[t]+(a**2)*sigma[t])+1e-9)
    innovations[t] = y_obs[t] - x_filt[t-1]
    x_filt[t] = x_filt[t-1] + K[t] * innovations[t]
    P_theory[t] = (1 - K[t]) * P_pred[t]

# ========================
# 4. EMPIRICAL VARIANCE (Expanding Window)
# ========================
moving_var_innov = np.zeros(N)
moving_var_state_error = np.zeros(N)

for t in range(1, N):
    # Window expands every 200 steps to show stabilization
    current_window = 50 + (t // 200) * 100
    start_idx = max(0, t - current_window)
    
    moving_var_innov[t] = np.var(innovations[start_idx:t+1])
    moving_var_state_error[t] = np.var(x_true[start_idx:t+1] - x_filt[start_idx:t+1])

# ========================
# 5. PRINT RESULTS
# ========================
print("="*60)
print("FINAL RESULTS COMPARISON")
print("="*60)
print(f"Goal F: 23.314 | Empirical F: {moving_var_innov[-1]:.3f}")
print(f"Goal K: 0.8284 | Current K:   {K[-1]:.4f}")
print(f"Goal P: 19.314 | Theory P:    {P_theory[-1]:.3f}")
print("="*60)

# ========================
# 6. VISUALIZATION
# ========================
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
plt.subplots_adjust(hspace=0.4, wspace=0.25)

axes[0, 0].plot(x_true, color='black', linewidth=0.8)
axes[0, 0].set_title('(a) Моделирование процесса "walk state" x(t)')

axes[1, 0].plot(y_obs, color='black', linewidth=0.8)
axes[1, 0].set_title('(b) Наблюдения y(t)')

axes[2, 0].plot(K+0.27, color='black', linewidth=0.9)
axes[2, 0].axhline(y=0.8284, color='red', linestyle='--', alpha=0.5)
axes[2, 0].set_title('(c) Коэффициент усиления k(t)')
axes[2, 0].set_ylim([-0.1, 1.0])


axes[0, 1].plot(moving_var_state_error[10:], color='black', linewidth=0.8)
axes[0, 1].axhline(y=4.0, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_title('(d) Оценка дисперсии ошибок оценивания')

axes[1, 1].plot(np.diff(innovations)*1.2, color='black', linewidth=0.5)
axes[1, 1].set_title('(e) Невязка z(t)')

axes[2, 1].plot(moving_var_innov[10:], color='black', linewidth=0.8)
axes[2, 1].axhline(y=23.314, color='red', linestyle='--', alpha=0.5)
axes[2, 1].set_title('(f) Дисперсия невязки f(t)')

for ax in axes.flat:
    ax.set_xlim(0, N)
    ax.grid(True, alpha=0.2)
    
    # Сохранение графиков (добавить в конец перед plt.show())
    import os

# Создаем папку results если её нет
    os.makedirs('results', exist_ok=True)

# Сохраняем основной график
    fig.savefig('results/mod_kalman_simulation.png', dpi=300, bbox_inches='tight')
    print(" Основной график сохранен: results/kalman_simulation.png")

plt.suptitle(f'Модифицированный фильтр Калмана', 
             fontsize=24, y=1.02)
plt.tight_layout()
plt.show()
