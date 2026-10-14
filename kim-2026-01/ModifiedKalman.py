# -*- coding: utf-8 -*-
"""
Created on Jan 21 2026
@author: K.N. Osipov (Modifications by Student)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ========================
# 1. PARAMETERS
# ========================
np.random.seed(42)
N = 2000

sigma_eta2 = 1.00  # Истинный шум процесса (W)
sigma_nu2 = 4   # Истинный шум измерений (V)
a = 0.99           # Коэффициент авторегрессии (стационарный случай)

# --- Строгое аналитическое решение Риккати для оптимального фильтра ---
# Квадратное уравнение для априорной дисперсии: P^2 + (r*(1-a^2) - q)*P - q*r = 0
coeff_b = sigma_nu2 * (1 - a**2) - sigma_eta2
P_opt_pred_analytical = (-coeff_b + np.sqrt(coeff_b**2 - 4 * 1 * (-sigma_eta2 * sigma_nu2))) / 2
K_opt_analytical = P_opt_pred_analytical / (P_opt_pred_analytical + sigma_nu2)
P_opt_post_analytical = (1 - K_opt_analytical) * P_opt_pred_analytical
F_opt_analytical = P_opt_pred_analytical + sigma_nu2

# ========================
# 2. DATA SIMULATION
# ========================
eta = np.random.normal(0, np.sqrt(sigma_eta2), N)
nu = np.random.normal(0, np.sqrt(sigma_nu2), N)
x_true = np.zeros(N)
for t in range(1, N):
    x_true[t] = a * x_true[t-1] + eta[t]
y_obs = x_true + nu

# ========================
# 3. FILTERS RUN
# ========================
# --- Модифицированный фильтр Осипова (Субоптимальный) ---
x_filt_sub = np.zeros(N)
sigma = np.zeros(N)
T = np.zeros(N)
K_sub = np.zeros(N)
innov_sub = np.zeros(N)

x_filt_sub[0] = y_obs[0]
sigma[0] = 10
T[0] = 10
kk = 1

# --- Строго оптимальный фильтр Калмана ---
x_filt_opt = np.zeros(N)
P_opt_theory = np.zeros(N)
P_opt_pred = np.zeros(N)
K_opt = np.zeros(N)
innov_opt = np.zeros(N)

x_filt_opt[0] = y_obs[0]
P_opt_theory[0] = P_opt_post_analytical  # инициализируем сразу в стационарном режиме

for t in range(1, N):
    # А. Расчет субоптимального фильтра Осипова
    sigma[t] = np.var(x_filt_sub[0:t]) + 1e-9
    T[t] = np.var(y_obs[0:t])
    K_sub[t] = ((kk * (a**2) * sigma[t] / ((T[t]) + (a**2) * sigma[t]) + 1e-9))
    
    innov_sub[t] = y_obs[t] - x_filt_sub[t-1]
    x_filt_sub[t] = x_filt_sub[t-1] + K_sub[t] * innov_sub[t]
    
    # Б. Расчет оптимального фильтра Калмана (знает истинные шумы)
    P_opt_pred[t] = (a**2) * P_opt_theory[t-1] + sigma_eta2
    K_opt[t] = P_opt_pred[t] / (P_opt_pred[t] + sigma_nu2)
    innov_opt[t] = y_obs[t] - a * x_filt_opt[t-1]
    x_filt_opt[t] = a * x_filt_opt[t-1] + K_opt[t] * innov_opt[t]
    P_opt_theory[t] = (1 - K_opt[t]) * P_opt_pred[t]

# ========================
# 4. EMPIRICAL VARIANCE (Expanding Window)
# ========================
moving_var_innov_sub = np.zeros(N)
moving_var_error_sub = np.zeros(N)
moving_var_error_opt = np.zeros(N)

for t in range(1, N):
    current_window = 50 + (t // 200) * 100
    start_idx = max(0, t - current_window)
    
    moving_var_innov_sub[t] = np.var(innov_sub[start_idx:t+1])
    moving_var_error_sub[t] = np.var(x_true[start_idx:t+1] - x_filt_sub[start_idx:t+1])
    moving_var_error_opt[t] = np.var(x_true[start_idx:t+1] - x_filt_opt[start_idx:t+1])

# ========================
# 5. PRINT RESULTS
# ========================
eta_index = moving_var_error_sub[-1] / moving_var_error_opt[-1]

print("="*60)
print("СТАТИСТИЧЕСКОЕ СОПОСТАВЛЕНИЕ РЕЗУЛЬТАТОВ (Установившийся режим)")
print("="*60)
print(f"Теоретический оптимальный K_opt: {K_opt_analytical:.4f} | Эмпирический K_sub:  {K_sub[-1]:.4f}")
print(f"Теоретическая оптимальная P_opt: {P_opt_post_analytical:.4f} | Реальная ошибка P_sub: {moving_var_error_sub[-1]:.4f}")
print(f"Теоретическая дисперсия невязки: {F_opt_analytical:.4f} | Реальная дисперсия:    {moving_var_innov_sub[-1]:.3f}")
print("-"*60)
print(f"РАСЧЕТНЫЙ ИНДЕКС СУБОПТИМАЛЬНОСТИ ЭВМ (eta = P_sub / P_opt): {eta_index:.4f}")
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

# График коэффициентов усиления (Сравнение)
axes[2, 0].plot(K_sub, color='black', linewidth=0.9, label='Эмпирический K_sub')
axes[2, 0].plot(K_opt, color='blue', linewidth=0.8, linestyle='--', label='Оптимальный K_opt')
axes[2, 0].axhline(y=K_opt_analytical, color='red', linestyle=':', alpha=0.7, label='Аналитический оптимум')
axes[2, 0].set_title('(c) Сравнение коэффициентов усиления k(t)')
axes[2, 0].set_ylim([-0.1, 1.2])
axes[2, 0].legend(loc='upper right')

# График дисперсии ошибок оценивания (Сравнение)
axes[0, 1].plot(moving_var_error_sub[10:], color='black', linewidth=0.9, label='Реальная ошибка P_sub')
axes[0, 1].plot(moving_var_error_opt[10:], color='blue', linewidth=0.8, linestyle='--', label='Оптимальная ошибка P_opt')
axes[0, 1].axhline(y=P_opt_post_analytical, color='red', linestyle=':', alpha=0.7, label='Теоретический минимум')
axes[0, 1].set_title('(d) Сравнение дисперсии ошибок оценивания')
axes[0, 1].legend(loc='upper right')

axes[1, 1].plot(np.diff(innov_sub)*1.2, color='black', linewidth=0.5)
axes[1, 1].set_title('(e) Невязка z(t)')

# График дисперсии невязок
axes[2, 1].plot(moving_var_innov_sub[10:], color='black', linewidth=0.8, label='Эмпирическая дисперсия невязки')
axes[2, 1].axhline(y=F_opt_analytical, color='red', linestyle='--', alpha=0.7, label='Теоретический оптимум')
axes[2, 1].set_title('(f) Дисперсия невязки f(t)')
axes[2, 1].legend(loc='upper right')

for ax in axes.flat:
    ax.set_xlim(0, N)
    ax.grid(True, alpha=0.2)

os.makedirs('results', exist_ok=True)
fig.savefig('results/mod_kalman_simulation.png', dpi=300, bbox_inches='tight')
print(" Основной график сохранен: results/mod_kalman_simulation.png")

plt.suptitle(f'Сравнительный анализ субоптимальности фильтра Калмана', fontsize=20, y=0.98)
plt.tight_layout()
plt.show()
