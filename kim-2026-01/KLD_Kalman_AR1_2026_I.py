# -*- coding: utf-8 -*-
"""
Честный KLD-фильтр Калмана для скалярного AR(1).

Воспроизводит 6 графиков из статьи:
    (a) Моделирование процесса "walk state" x(t)
    (b) Наблюдения y(t)
    (c) Коэффициент усиления k(t)
    (d) Оценка дисперсии ошибок оценивания
    (e) Невязка z(t)
    (f) Дисперсия невязки f(t)

Отличие от статьи: используется честный KLD-вывод
    R_hat = Var(nu) - P^-
    k = P^- / (P^- + R_hat)
а не эвристическая формула с k0.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. ПАРАМЕТРЫ МОДЕЛИ
# ============================================================
np.random.seed(42)
N = 2000

# Значения из статьи (для воспроизведения графиков)
sigma_eta2 = 16.00   # Q — дисперсия шума процесса
sigma_nu2  = 4.0     # R — дисперсия шума измерения
a = 1.0              # коэффициент AR(1) — случайное блуждание

# ============================================================
# 2. ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================
eta = np.random.normal(0, np.sqrt(sigma_eta2), N)
nu  = np.random.normal(0, np.sqrt(sigma_nu2), N)

x_true = np.zeros(N)
for t in range(1, N):
    x_true[t] = a * x_true[t-1] + eta[t]

y_obs = x_true + nu

# ============================================================
# 3. ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ (уравнение Риккати для a=1)
# ============================================================
# P^2 + Q*P - R*Q = 0
P_theory = (-sigma_eta2 + np.sqrt(sigma_eta2**2 + 4*sigma_nu2*sigma_eta2)) / 2
P_pred_theory = P_theory + sigma_eta2
K_theory = P_pred_theory / (P_pred_theory + sigma_nu2)

# Дисперсия невязки
F_theory = P_pred_theory + sigma_nu2

print("=" * 60)
print("ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ (a = 1)")
print("=" * 60)
print(f"P (апостериорная)  = {P_theory:.4f}")
print(f"P^- (априорная)    = {P_pred_theory:.4f}")
print(f"K (коэффициент)    = {K_theory:.4f}")
print(f"F = Var(nu)        = {F_theory:.4f}")
print("=" * 60)

# ============================================================
# 4. ЧЕСТНЫЙ KLD-ФИЛЬТР
# ============================================================
x_kld = np.zeros(N)
P_kld = np.zeros(N)
K_kld = np.zeros(N)
nu_all = np.zeros(N)
R_hat_arr = np.zeros(N)
var_nu_arr = np.zeros(N)

# Начальные условия
x_kld[0] = y_obs[0]
P_kld[0] = sigma_nu2
R_hat_arr[0] = sigma_nu2

window = 50   # окно усреднения невязки

for t in range(1, N):
    # --- Predict ---
    x_pred = a * x_kld[t-1]
    P_pred = a**2 * P_kld[t-1] + sigma_eta2

    # --- Innovation ---
    nu_all[t] = y_obs[t] - x_pred

    # --- KLD-минимизация: R_hat = Var(nu) - P^- ---
    if t >= window:
        var_nu = np.var(nu_all[t-window+1:t+1])
        R_hat_arr[t] = max(var_nu - P_pred, 1e-6)
    else:
        var_nu = P_pred + R_hat_arr[t-1]
        R_hat_arr[t] = R_hat_arr[t-1]
    var_nu_arr[t] = var_nu

    # --- KLD-формула для k ---
    # k = P^- / (P^- + R_hat) = P^- / Var(nu)
    K_kld[t] = P_pred / (P_pred + R_hat_arr[t])

    # --- Update ---
    x_kld[t] = x_pred + K_kld[t] * nu_all[t]
    P_kld[t] = (1 - K_kld[t]) * P_pred

# ============================================================
# 5. ЭМПИРИЧЕСКИЕ ДИСПЕРСИИ (скользящее окно)
# ============================================================
moving_var_innov = np.zeros(N)
moving_var_state_error = np.zeros(N)

for t in range(1, N):
    current_window = 50 + (t // 200) * 100
    start_idx = max(0, t - current_window)

    moving_var_innov[t] = np.var(nu_all[start_idx:t+1])
    moving_var_state_error[t] = np.var(x_true[start_idx:t+1] - x_kld[start_idx:t+1])

# ============================================================
# 6. ПЕЧАТЬ РЕЗУЛЬТАТОВ
# ============================================================
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ЧЕСТНОГО KLD-ФИЛЬТРА")
print("=" * 60)
print(f"Goal F: {F_theory:.3f} | Empirical F: {moving_var_innov[-1]:.3f}")
print(f"Goal K: {K_theory:.4f} | Current K:   {K_kld[-1]:.4f}")
print(f"Goal P: {P_theory:.3f} | Theory P:    {P_kld[-1]:.3f}")
print(f"R истинное: {sigma_nu2:.3f} | R̂ финальное: {R_hat_arr[-1]:.3f}")
print("=" * 60)

# ============================================================
# 7. ГРАФИКИ (6 подграфиков, как в статье)
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
plt.subplots_adjust(hspace=0.4, wspace=0.25)

# (a) Моделирование процесса "walk state" x(t)
axes[0, 0].plot(x_true, color='black', linewidth=0.8)
axes[0, 0].set_title('(a) Моделирование процесса "walk state" x(t)')
axes[0, 0].set_xlim(0, N)
axes[0, 0].grid(True, alpha=0.2)

# (b) Наблюдения y(t)
axes[1, 0].plot(y_obs, color='black', linewidth=0.8)
axes[1, 0].set_title('(b) Наблюдения y(t)')
axes[1, 0].set_xlim(0, N)
axes[1, 0].grid(True, alpha=0.2)

# (c) Коэффициент усиления k(t)
axes[2, 0].plot(K_kld, color='black', linewidth=0.9)
axes[2, 0].axhline(y=K_theory, color='red', linestyle='--', alpha=0.7,
                   label=f'Теория K = {K_theory:.4f}')
axes[2, 0].set_title('(c) Коэффициент усиления k(t)')
axes[2, 0].set_ylim([-0.1, 1.0])
axes[2, 0].set_xlim(0, N)
axes[2, 0].grid(True, alpha=0.2)
axes[2, 0].legend(fontsize=9)

# (d) Оценка дисперсии ошибок оценивания
axes[0, 1].plot(moving_var_state_error[10:], color='black', linewidth=0.8)
axes[0, 1].axhline(y=P_theory, color='red', linestyle='--', alpha=0.7,
                   label=f'Теория P = {P_theory:.3f}')
axes[0, 1].set_title('(d) Оценка дисперсии ошибок оценивания')
axes[0, 1].set_xlim(0, N)
axes[0, 1].grid(True, alpha=0.2)
axes[0, 1].legend(fontsize=9)

# (e) Невязка z(t)
axes[1, 1].plot(nu_all, color='black', linewidth=0.5)
axes[1, 1].set_title('(e) Невязка z(t)')
axes[1, 1].set_xlim(0, N)
axes[1, 1].grid(True, alpha=0.2)

# (f) Дисперсия невязки f(t)
axes[2, 1].plot(moving_var_innov[10:], color='black', linewidth=0.8)
axes[2, 1].axhline(y=F_theory, color='red', linestyle='--', alpha=0.7,
                   label=f'Теория F = {F_theory:.3f}')
axes[2, 1].set_title('(f) Дисперсия невязки f(t)')
axes[2, 1].set_xlim(0, N)
axes[2, 1].grid(True, alpha=0.2)
axes[2, 1].legend(fontsize=9)

plt.suptitle('Честный KLD-фильтр Калмана', fontsize=20, y=1.00)
plt.tight_layout()

# Сохранение
os.makedirs('results', exist_ok=True)
fig.savefig('results/kld_kalman_full.png', dpi=200, bbox_inches='tight')
print("\nГрафик сохранён: results/kld_kalman_full.png")

plt.show()

# ============================================================
# 8. ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: K(t) детально
# ============================================================
fig2, ax = plt.subplots(figsize=(12, 6))

ax.plot(K_kld, color='red', lw=1.2, label='KLD-фильтр (честный)')
ax.axhline(y=K_theory, color='black', ls=':', lw=2,
           label=f'Теория K = {K_theory:.4f}')

ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('K(t)', fontsize=12)
ax.set_title('Коэффициент усиления K(t) — честный KLD', fontsize=14)
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
fig2.savefig('results/kld_gain_K.png', dpi=200, bbox_inches='tight')
print("График K(t) сохранён: results/kld_gain_K.png")

plt.show()