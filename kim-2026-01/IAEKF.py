# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:47:53 2026

@author: assis
"""

# -*- coding: utf-8 -*-
"""
IAEKF для скалярного AR(1) — Innovation-based Adaptive Kalman Filter.

Модель:
    x_t = a * x_{t-1} + w_t,   w_t ~ N(0, Q)
    y_t = x_t + v_t,           v_t ~ N(0, R)

Адаптация R:
    nu_t = y_t - a * x_hat_{t-1}
    C_v  = Var(nu)              (эмпирическая дисперсия невязки)
    R_hat = C_v - P^-
    K    = P^- / (P^- + R_hat) = P^- / C_v
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. ПАРАМЕТРЫ МОДЕЛИ
# ============================================================
np.random.seed(42)

N      = 2000
a      = 0.99        # коэффициент AR(1)
Q      = 1.0         # дисперсия шума процесса
R_true = 4.0         # истинная дисперсия шума измерения
window = 50          # размер окна усреднения невязки

# ============================================================
# 2. ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================
w = np.random.normal(0, np.sqrt(Q), N)
v = np.random.normal(0, np.sqrt(R_true), N)

x_true = np.zeros(N)
x_true[0] = np.random.normal(0, np.sqrt(Q / (1 - a**2)))
for t in range(1, N):
    x_true[t] = a * x_true[t-1] + w[t]

y_obs = x_true + v

# ============================================================
# 3. ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ (уравнение Риккати)
# ============================================================
b_coef = Q + R_true * (1 - a**2)
disc   = b_coef**2 + 4 * a**2 * R_true * Q
P_theory      = (-b_coef + np.sqrt(disc)) / (2 * a**2)
P_pred_theory = a**2 * P_theory + Q
K_theory      = P_pred_theory / (P_pred_theory + R_true)

print("=" * 60)
print("ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ")
print("=" * 60)
print(f"P  (апостериорная)   = {P_theory:.4f}")
print(f"P^- (априорная)      = {P_pred_theory:.4f}")
print(f"K  (коэффициент)     = {K_theory:.4f}")
print("=" * 60)

# ============================================================
# 4. КЛАССИЧЕСКИЙ ФИЛЬТР (эталон, знаем Q и R)
# ============================================================
x_classic = np.zeros(N)
P_classic = np.zeros(N)
K_classic = np.zeros(N)

x_classic[0] = y_obs[0]
P_classic[0] = R_true

for t in range(1, N):
    # Predict
    x_pred = a * x_classic[t-1]
    P_pred = a**2 * P_classic[t-1] + Q

    # Update
    K_classic[t] = P_pred / (P_pred + R_true)
    x_classic[t] = x_pred + K_classic[t] * (y_obs[t] - x_pred)
    P_classic[t] = (1 - K_classic[t]) * P_pred

# ============================================================
# 5. IAEKF — АДАПТИВНЫЙ ФИЛЬТР
# ============================================================
x_adapt = np.zeros(N)
P_adapt = np.zeros(N)
K_adapt = np.zeros(N)
nu_all  = np.zeros(N)
R_hat   = np.zeros(N)
C_v_arr = np.zeros(N)

# Начальные условия
x_adapt[0] = y_obs[0]
P_adapt[0] = R_true
R_hat[0]   = R_true

for t in range(1, N):
    # --------------------------------------------------------
    # Шаг 1. Predict
    # --------------------------------------------------------
    x_pred = a * x_adapt[t-1]
    P_pred = a**2 * P_adapt[t-1] + Q

    # --------------------------------------------------------
    # Шаг 2. Innovation
    # --------------------------------------------------------
    nu_all[t] = y_obs[t] - x_pred

    # --------------------------------------------------------
    # Шаг 3. Эмпирическая дисперсия невязки
    # --------------------------------------------------------
    if t >= window:
        C_v = np.var(nu_all[t-window+1:t+1])
    else:
        C_v = P_pred + R_hat[t-1]
    C_v_arr[t] = C_v

    # --------------------------------------------------------
    # Шаг 4. Адаптивная оценка R
    # --------------------------------------------------------
    # C_v = P^- + R  =>  R_hat = C_v - P^-
    R_hat_new = C_v - P_pred

    # Защита: R должна быть положительной
    R_hat[t] = max(R_hat_new, 1e-6)

    # --------------------------------------------------------
    # Шаг 5. Коэффициент усиления
    # --------------------------------------------------------
    K_adapt[t] = P_pred / (P_pred + R_hat[t])

    # --------------------------------------------------------
    # Шаг 6. Update
    # --------------------------------------------------------
    x_adapt[t] = x_pred + K_adapt[t] * nu_all[t]
    P_adapt[t] = (1 - K_adapt[t]) * P_pred

# ============================================================
# 6. МЕТРИКИ
# ============================================================
rmse_classic = np.sqrt(np.mean((x_true - x_classic)**2))
rmse_adapt   = np.sqrt(np.mean((x_true - x_adapt)**2))

K_adapt_mean   = np.mean(K_adapt[-500:])
R_hat_mean     = np.mean(R_hat[-500:])
P_adapt_mean   = np.mean(P_adapt[-500:])

print("\n" + "=" * 60)
print("СРАВНЕНИЕ ФИЛЬТРОВ")
print("=" * 60)
print(f"RMSE (классический)   = {rmse_classic:.4f}")
print(f"RMSE (IAEKF)          = {rmse_adapt:.4f}")
print(f"K теория              = {K_theory:.4f}")
print(f"K классический (сред.)= {np.mean(K_classic[-500:]):.4f}")
print(f"K IAEKF (сред.)       = {K_adapt_mean:.4f}   "
      f"(откл. {abs(K_adapt_mean-K_theory)/K_theory*100:.2f}%)")
print(f"P теория              = {P_theory:.4f}")
print(f"P IAEKF (сред.)       = {P_adapt_mean:.4f}")
print(f"R истинное            = {R_true:.4f}")
print(f"R IAEKF (сред.)       = {R_hat_mean:.4f}   "
      f"(откл. {abs(R_hat_mean-R_true)/R_true*100:.2f}%)")
print("=" * 60)

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.25)

# (a) Истина и наблюдения
axes[0, 0].plot(x_true, 'k', lw=0.8, label='x_true')
axes[0, 0].plot(y_obs, 'gray', lw=0.4, alpha=0.5, label='y_obs')
axes[0, 0].set_title('(a) Истинный процесс и наблюдения')
axes[0, 0].legend(fontsize=8)

# (b) Оценки
axes[0, 1].plot(x_true, 'k', lw=0.8, label='x_true')
axes[0, 1].plot(x_classic, 'b', lw=0.8, label='классический')
axes[0, 1].plot(x_adapt, 'r--', lw=0.8, label='IAEKF')
axes[0, 1].set_title('(b) Оценки состояния')
axes[0, 1].legend(fontsize=8)

# (c) K(t)
axes[1, 0].plot(K_classic, 'b', lw=0.9, label='классический')
axes[1, 0].plot(K_adapt, 'r--', lw=0.9, label='IAEKF')
axes[1, 0].axhline(y=K_theory, color='k', ls=':', lw=1.5,
                   label=f'теория K = {K_theory:.4f}')
axes[1, 0].set_title('(c) Коэффициент усиления K(t)')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].legend(fontsize=8)

# (d) P(t)
axes[1, 1].plot(P_classic, 'b', lw=0.9, label='классический')
axes[1, 1].plot(P_adapt, 'r--', lw=0.9, label='IAEKF')
axes[1, 1].axhline(y=P_theory, color='k', ls=':', lw=1.5,
                   label=f'теория P = {P_theory:.4f}')
axes[1, 1].set_title('(d) Апостериорная дисперсия P(t)')
axes[1, 1].legend(fontsize=8)

# (e) Невязка
axes[2, 0].plot(nu_all, 'k', lw=0.5)
axes[2, 0].set_title('(e) Невязка ν(t)')
axes[2, 0].set_xlabel('t')

# (f) R_hat и Var(nu)
axes[2, 1].plot(R_hat, 'r', lw=0.9, label='R̂ = Var(ν) - P^-')
axes[2, 1].plot(C_v_arr, 'gray', lw=0.6, alpha=0.7, label='Var(ν)')
axes[2, 1].axhline(y=R_true, color='k', ls=':', lw=1.5,
                   label=f'истина R = {R_true}')
axes[2, 1].set_title('(f) Адаптивная оценка R')
axes[2, 1].set_xlabel('t')
axes[2, 1].legend(fontsize=8)

for ax in axes.flat:
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, N)

plt.suptitle('IAEKF для скалярного AR(1)', fontsize=16, y=1.00)
plt.tight_layout()

os.makedirs('results', exist_ok=True)
fig.savefig('results/iaekf_ar1.png', dpi=200, bbox_inches='tight')
print("\nГрафик сохранён: results/iaekf_ar1.png")
plt.show()

# ============================================================
# 8. ТЕСТ НА РАЗНЫХ ПАРАМЕТРАХ
# ============================================================
print("\n" + "=" * 60)
print("ТЕСТ НА РАЗНЫХ ПАРАМЕТРАХ")
print("=" * 60)

test_cases = [
    # (a, Q, R)
    (0.95, 1.0, 4.0),
    (0.90, 0.1, 1.0),
    (0.99, 16.0, 4.0),
    (1.00, 16.0, 4.0),
    (0.50, 1.0, 1.0),
]

print(f"{'a':>5} {'Q':>6} {'R':>6} | "
      f"{'K_theory':>10} {'K_IAEKF':>10} {'R_IAEKF':>10} | "
      f"{'откл K, %':>10} {'откл R, %':>10}")
print("-" * 80)

for (a_t, Q_t, R_t) in test_cases:
    # Генерация
    np.random.seed(42)
    w_t = np.random.normal(0, np.sqrt(Q_t), N)
    v_t = np.random.normal(0, np.sqrt(R_t), N)

    x_t = np.zeros(N)
    if a_t < 1:
        x_t[0] = np.random.normal(0, np.sqrt(Q_t / (1 - a_t**2)))
    for i in range(1, N):
        x_t[i] = a_t * x_t[i-1] + w_t[i]
    y_t = x_t + v_t

    # Теория
    if a_t < 1:
        b = Q_t + R_t * (1 - a_t**2)
        d = b**2 + 4 * a_t**2 * R_t * Q_t
        P_th = (-b + np.sqrt(d)) / (2 * a_t**2)
        P_pred_th = a_t**2 * P_th + Q_t
        K_th = P_pred_th / (P_pred_th + R_t)
    else:
        P_th = (-Q_t + np.sqrt(Q_t**2 + 4 * R_t * Q_t)) / 2
        P_pred_th = P_th + Q_t
        K_th = P_pred_th / (P_pred_th + R_t)

    # IAEKF
    x_ad = np.zeros(N)
    P_ad = np.zeros(N)
    K_ad = np.zeros(N)
    nu_ad = np.zeros(N)
    R_ad = np.zeros(N)
    x_ad[0] = y_t[0]
    P_ad[0] = R_t
    R_ad[0] = R_t

    for i in range(1, N):
        x_pr = a_t * x_ad[i-1]
        P_pr = a_t**2 * P_ad[i-1] + Q_t
        nu_ad[i] = y_t[i] - x_pr
        if i >= window:
            Cv = np.var(nu_ad[i-window+1:i+1])
        else:
            Cv = P_pr + R_ad[i-1]
        R_ad[i] = max(Cv - P_pr, 1e-6)
        K_ad[i] = P_pr / (P_pr + R_ad[i])
        x_ad[i] = x_pr + K_ad[i] * nu_ad[i]
        P_ad[i] = (1 - K_ad[i]) * P_pr

    K_mean = np.mean(K_ad[-500:])
    R_mean = np.mean(R_ad[-500:])
    err_K = abs(K_mean - K_th) / K_th * 100 if K_th > 0 else 0
    err_R = abs(R_mean - R_t) / R_t * 100

    print(f"{a_t:>5.2f} {Q_t:>6.2f} {R_t:>6.2f} | "
          f"{K_th:>10.4f} {K_mean:>10.4f} {R_mean:>10.4f} | "
          f"{err_K:>10.2f} {err_R:>10.2f}")
print("=" * 80)