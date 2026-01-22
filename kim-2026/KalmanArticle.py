# -*- coding: utf-8 -*-
"""
Created on Jan 21 2026
@author: K.N. Osipov
"""

import numpy as np
import matplotlib.pyplot as plt

# ========================
# 1. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ========================
np.random.seed(42)
N = 2000
sigma_eta2 = 16.0  # Шум процесса (W, Q)
sigma_nu2 = 4.0    # Шум измерений (V, R)

# ========================
# 2. МОДЕЛИРОВАНИЕ ДАННЫХ
# ========================
eta = np.random.normal(0, np.sqrt(sigma_eta2), N)
nu = np.random.normal(0, np.sqrt(sigma_nu2), N)

x_true = np.zeros(N)
for t in range(1, N):
    x_true[t] = x_true[t-1] + eta[t]
y_obs = x_true + nu

# ========================
# 3. ФИЛЬТР КАЛМАНА
# ========================
x_filt = np.zeros(N)
P_theory = np.zeros(N)
K = np.zeros(N)
innovations = np.zeros(N)

x_filt[0] = y_obs[0]
P_theory[0] = 1.0 

for t in range(1, N):
    
    P_pred = P_theory[t-1] + sigma_eta2
    F_t = P_pred + sigma_nu2
    K[t] = P_pred / F_t
    innovations[t] = y_obs[t] - x_filt[t-1]
    x_filt[t] = x_filt[t-1] + K[t] * innovations[t]
    P_theory[t] = (1 - K[t]) * P_pred

# ========================
# 4. РАСЧЕТ ДИСПЕРСИИ 
# ========================
moving_var_innov = np.zeros(N)
moving_var_state_error = np.zeros(N)

for t in range(1, N):
    current_window = 50 + (t // 200) * 100
    start_idx = max(0, t - current_window)
    
    moving_var_innov[t] = np.var(innovations[start_idx:t+1])
    moving_var_state_error[t] = np.var(x_true[start_idx:t+1] - x_filt[start_idx:t+1])

# ========================
# 5. ТЕОРЕТИЧЕСКИЙ РАСЧЕТ 
# ========================
# Уравнение Риккати для стационарного состояния:
# P² - Q*P - Q*R = 0
D = sigma_eta2**2 + 4 * sigma_eta2 * sigma_nu2
P_steady_theory = (sigma_eta2 + np.sqrt(D)) / 2  # Положительный корень!
F_steady_theory = P_steady_theory + sigma_nu2   # sigma_nu2, а не sigma_eta2!
K_steady_theory = P_steady_theory / F_steady_theory


# ========================
# 6. РАСПЕЧАТКА РЕЗУЛЬТАТОВ (НОВЫЙ БЛОК)
# ========================
print("="*60)
print("KALMAN FILTER RESULTS (Steady State Analysis)")
print("="*60)
print(f"Parameters: Sigma_eta^2 (W) = {sigma_eta2}, Sigma_nu^2 (V) = {sigma_nu2}")
print("-"*60)
print(f"{'Metric':<30} | {'Theoretical':<12} | {'Empirical*':<12}")
print("-"*60)
print(f"{'State Variance (P)':<30} | {P_steady_theory:<12.4f} | {np.mean(P_theory[-100:]):<12.4f}")
print(f"{'Innovation Variance (F)':<30} | {F_steady_theory:<12.4f} | {moving_var_innov[-1]:<12.4f}")
print(f"{'Kalman Gain (K)':<30} | {K_steady_theory:<12.4f} | {K[-1]:<12.4f}")
print("-"*60)
print("*Empirical values calculated at t = 2000")
print("="*60)

# ========================
# 7. ВИЗУАЛИЗАЦИЯ
# ========================
fig, axes = plt.subplots(3, 2, figsize=(14, 11))
plt.subplots_adjust(hspace=0.4, wspace=0.25)

# (a) State
axes[0, 0].plot(x_true, color='black', linewidth=0.8)
axes[0, 0].set_title('(a) Моделирование процесса "walk state" x(t)')

# (b) Observations
axes[1, 0].plot(y_obs, color='black', linewidth=0.8)
axes[1, 0].set_title('(b) Моделирование процесса наблюдения y(t)')

# (c) Kalman Gain
axes[2, 0].plot(K, color='black', linewidth=0.9)
axes[2, 0].axhline(y=K_steady_theory, color='red', linestyle='--', alpha=0.5)
axes[2, 0].set_title('(c) Коэффициент усиления k(t)')
axes[2, 0].set_ylim(0, 1.0)

# (d) State Variance
axes[0, 1].plot(moving_var_state_error[10:], color='black', linewidth=0.8)
axes[0, 1].axhline(y=sigma_nu2, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_title('(d) Оценка дисперсии ошибок оценивания вектора состояния x(t)')

# (e) Innovations
axes[1, 1].plot(innovations, color='black', linewidth=0.5)
axes[1, 1].set_title('(e) Инновация (невязка) z(t)')

# (f) Innovation Variance
axes[2, 1].plot(moving_var_innov[10:], color='black', linewidth=0.8)
axes[2, 1].axhline(y=F_steady_theory, color='red', linestyle='--', alpha=0.5)
axes[2, 1].set_title('(f) Выборочная оценка дисперсии невязки')

for ax in axes.flat:
    ax.set_xlim(0, N)
    ax.grid(True, alpha=0.2)
    
    
    # Сохранение графиков (добавить в конец перед plt.show())
    import os

# Создаем папку results если её нет
    os.makedirs('results', exist_ok=True)

# Сохраняем основной график
    fig.savefig('results/kalman_simulation.png', dpi=300, bbox_inches='tight')
    print("✅ Основной график сохранен: results/kalman_simulation.png")

# Сохраняем статистический анализ
    fig.savefig('results/innovations_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ График анализа сохранен: results/innovations_analysis.png")

    plt.show()


from statsmodels.graphics.tsaplots import plot_acf

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))

axes2[0].hist(innovations, color='black', linewidth=0.8)
axes2[0].set_title('(a) Гистограмма невязок')

# График автокорреляционной функции с доверительным интервалом
plot_acf(innovations[2:], lags=8, ax=axes2[1], color='black', 
         title='(b) Автокорреляция невязок', 
         alpha=0.2)
axes2[1].set_xlabel('Лаг')
axes2[1].set_ylabel('Автокорреляция')
axes2[1].set_ylim(-0.3, 0.8)
axes2[1].set_ylim(-0.3, 1.1)  # ← Увеличить до 1.1

# Сохранение графиков (добавить в конец перед plt.show())
import os

# Создаем папку results если её нет
os.makedirs('results', exist_ok=True)

# Сохраняем основной график
fig.savefig('results/kalman_simulation.png', dpi=300, bbox_inches='tight')
print("✅ Основной график сохранен: results/kalman_simulation.png")

# Сохраняем статистический анализ
fig2.savefig('results/innovations_analysis.png', dpi=300, bbox_inches='tight')
print("✅ График анализа сохранен: results/innovations_analysis.png")

plt.show()