# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:20:56 2026

@author: assis
"""

# -*- coding: utf-8 -*-
"""
PD-Kalman Filter для скалярного AR(1).

Идея: к стандартному обновлению Калмана добавить производную невязки.

    x̂_t = a·x̂_{t-1} + K_t·ν_t + D_t·ν̇_t

где:
    ν_t = y_t - a·x̂_{t-1}       — невязка (innovation)
    ν̇_t = ν_t - ν_{t-1}          — производная невязки
    K_t                          — коэффициент Калмана (оптимальный)
    D_t                          — коэффициент при производной (адаптивный)

Сравнение:
    - классический Калман (D = 0)
    - PD-Калман с фиксированным D
    - PD-Калман с адаптивным D через KLD
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================
np.random.seed(42)
N = 2000
a = 0.95
Q = 1.0

# R скачком меняется — тест на нестационарность
R_segments = [
    (0,    500,  1.0),
    (500,  1000, 4.0),
    (1000, 1500, 0.5),
    (1500, N,    4.0),
]

eps = 1e-6

# ============================================================
# 2. ДАННЫЕ
# ============================================================
w_noise = np.random.normal(0, np.sqrt(Q), N)
v_noise = np.zeros(N)
R_true  = np.zeros(N)

for (t0, t1, R_val) in R_segments:
    v_noise[t0:t1] = np.random.normal(0, np.sqrt(R_val), t1 - t0)
    R_true[t0:t1]  = R_val

x_true = np.zeros(N)
x_true[0] = np.random.normal(0, np.sqrt(Q / (1 - a**2)))
for t in range(1, N):
    x_true[t] = a * x_true[t-1] + w_noise[t]

y_obs = x_true + v_noise

# ============================================================
# 3. ФУНКЦИЯ KLD
# ============================================================
def kld_gaussian(var_emp, var_theor):
    if var_emp <= 0 or var_theor <= 0:
        return np.inf
    return 0.5 * (var_emp / var_theor + np.log(var_theor)
                  - np.log(var_emp) - 1.0)

# ============================================================
# 4. КЛАССИЧЕСКИЙ КАЛМАН (D = 0)
# ============================================================
def run_classic(y_obs, a, Q, R_init, window=50):
    N = len(y_obs)
    x = np.zeros(N)
    P = np.zeros(N)
    K = np.zeros(N)
    nu = np.zeros(N)
    R_hat = np.zeros(N)

    x[0] = y_obs[0]
    P[0] = R_init
    R_hat[0] = R_init

    for t in range(1, N):
        x_pred = a * x[t-1]
        P_pred = a**2 * P[t-1] + Q
        nu[t] = y_obs[t] - x_pred

        if t >= window:
            C_v = np.var(nu[t-window+1:t+1])
        else:
            C_v = P_pred + R_hat[t-1]

        R_hat[t] = max(C_v - P_pred, eps)
        K[t] = P_pred / (P_pred + R_hat[t])
        x[t] = x_pred + K[t] * nu[t]
        P[t] = (1 - K[t]) * P_pred

    return x, P, K, nu, R_hat

# ============================================================
# 5. PD-КАЛМАН с фиксированным D
# ============================================================
def run_pd_fixed(y_obs, a, Q, R_init, D, window=50):
    N = len(y_obs)
    x = np.zeros(N)
    P = np.zeros(N)
    K = np.zeros(N)
    nu = np.zeros(N)
    nu_dot = np.zeros(N)
    R_hat = np.zeros(N)

    x[0] = y_obs[0]
    P[0] = R_init
    R_hat[0] = R_init

    for t in range(1, N):
        x_pred = a * x[t-1]
        P_pred = a**2 * P[t-1] + Q
        nu[t] = y_obs[t] - x_pred
        nu_dot[t] = nu[t] - nu[t-1] if t > 1 else 0.0

        if t >= window:
            C_v = np.var(nu[t-window+1:t+1])
        else:
            C_v = P_pred + R_hat[t-1]

        R_hat[t] = max(C_v - P_pred, eps)
        K[t] = P_pred / (P_pred + R_hat[t])

        # PD-обновление
        x[t] = x_pred + K[t] * nu[t] + D * nu_dot[t]
        P[t] = (1 - K[t]) * P_pred

    return x, P, K, nu, nu_dot, R_hat

# ============================================================
# 6. PD-КАЛМАН с АДАПТИВНЫМ D через KLD
# ============================================================
def run_pd_adaptive(y_obs, a, Q, R_init, window=50,
                    D_candidates=np.linspace(0, 0.5, 11),
                    w_D=50):
    N = len(y_obs)
    x = np.zeros(N)
    P = np.zeros(N)
    K = np.zeros(N)
    D_arr = np.zeros(N)
    nu = np.zeros(N)
    nu_dot = np.zeros(N)
    R_hat = np.zeros(N)
    kld_arr = np.zeros(N)

    x[0] = y_obs[0]
    P[0] = R_init
    R_hat[0] = R_init
    D_arr[0] = 0.0

    for t in range(1, N):
        x_pred = a * x[t-1]
        P_pred = a**2 * P[t-1] + Q
        nu[t] = y_obs[t] - x_pred
        nu_dot[t] = nu[t] - nu[t-1] if t > 1 else 0.0

        if t >= window:
            C_v = np.var(nu[t-window+1:t+1])
        else:
            C_v = P_pred + R_hat[t-1]

        R_hat[t] = max(C_v - P_pred, eps)
        K[t] = P_pred / (P_pred + R_hat[t])

        # --------------------------------------------------------
        # Адаптивный выбор D через KLD
        # --------------------------------------------------------
        if t >= w_D:
            best_kld = np.inf
            best_D = 0.0

            for D_c in D_candidates:
                # Оценка с данным D
                x_test = np.zeros(w_D)
                x_test[0] = x[t - w_D]
                for j in range(1, w_D):
                    idx = t - w_D + j
                    x_test[j] = (a * x_test[j-1]
                                 + K[idx] * nu[idx]
                                 + D_c * nu_dot[idx])
                # Невязки для тестовой оценки
                nu_test = y_obs[t-w_D+1:t+1] - x_test
                if len(nu_test) < 2:
                    continue
                var_nu_test = np.var(nu_test)
                if var_nu_test <= 0:
                    continue
                J = kld_gaussian(var_nu_test, P_pred + R_hat[t])
                if J < best_kld:
                    best_kld = J
                    best_D = D_c

            D_arr[t] = best_D
            kld_arr[t] = best_kld
        else:
            D_arr[t] = 0.0
            kld_arr[t] = 0.0

        # PD-обновление с выбранным D
        x[t] = x_pred + K[t] * nu[t] + D_arr[t] * nu_dot[t]
        P[t] = (1 - K[t]) * P_pred

    return x, P, K, D_arr, nu, nu_dot, R_hat, kld_arr

# ============================================================
# 7. ЗАПУСК
# ============================================================
R_init = R_true[0]

x_cl, P_cl, K_cl, nu_cl, R_cl = run_classic(y_obs, a, Q, R_init)
x_pd, P_pd, K_pd, nu_pd, nu_dot_pd, R_pd = run_pd_fixed(
    y_obs, a, Q, R_init, D=0.2)
x_ad, P_ad, K_ad, D_ad, nu_ad, nu_dot_ad, R_ad, kld_ad = run_pd_adaptive(
    y_obs, a, Q, R_init)

# ============================================================
# 8. МЕТРИКИ
# ============================================================
def rmse(x_true, x_hat):
    return np.sqrt(np.mean((x_true - x_hat)**2))

def err_R(R_hat, R_true):
    return np.mean(np.abs(R_hat - R_true))

# Ошибка на скачках R
transition_mask = np.zeros(N, dtype=bool)
for (t0, t1, _) in R_segments[1:]:
    transition_mask[t0:t0+50] = True

def err_transition(x_true, x_hat, mask):
    return np.sqrt(np.mean((x_true[mask] - x_hat[mask])**2))

print("=" * 70)
print("СРАВНЕНИЕ: КЛАССИЧЕСКИЙ vs PD-КАЛМАН")
print("=" * 70)
print(f"{'Метод':<30} {'RMSE':>8} {'RMSE(скач)':>12} "
      f"{'Err R':>8}")
print("-" * 70)
print(f"{'Классический':<30} {rmse(x_true, x_cl):>8.4f} "
      f"{err_transition(x_true, x_cl, transition_mask):>12.4f} "
      f"{err_R(R_cl, R_true):>8.4f}")
print(f"{'PD-Калман (D=0.2)':<30} {rmse(x_true, x_pd):>8.4f} "
      f"{err_transition(x_true, x_pd, transition_mask):>12.4f} "
      f"{err_R(R_pd, R_true):>8.4f}")
print(f"{'PD-Калман (адапт. D)':<30} {rmse(x_true, x_ad):>8.4f} "
      f"{err_transition(x_true, x_ad, transition_mask):>12.4f} "
      f"{err_R(R_ad, R_true):>8.4f}")
print("=" * 70)

# Улучшение на скачках
rmse_cl_trans = err_transition(x_true, x_cl, transition_mask)
rmse_ad_trans = err_transition(x_true, x_ad, transition_mask)
print(f"\nУлучшение на скачках: "
      f"{(rmse_cl_trans - rmse_ad_trans)/rmse_cl_trans*100:.1f}%")

# ============================================================
# 9. ВИЗУАЛИЗАЦИЯ
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
axes[0, 1].plot(x_cl, 'b', lw=0.8, alpha=0.7, label='классический')
axes[0, 1].plot(x_ad, 'r--', lw=0.8, label='PD (адапт. D)')
axes[0, 1].set_title('(b) Оценки состояния')
axes[0, 1].legend(fontsize=8)

# (c) K(t)
axes[1, 0].plot(K_cl, 'b', lw=0.9, alpha=0.7, label='классический')
axes[1, 0].plot(K_ad, 'r--', lw=0.9, label='PD (адапт. D)')
axes[1, 0].set_title('(c) Коэффициент усиления K(t)')
axes[1, 0].legend(fontsize=8)

# (d) D(t)
axes[1, 1].plot(D_ad, 'r', lw=0.9)
axes[1, 1].set_title('(d) Адаптивный коэффициент D(t)')
axes[1, 1].set_xlabel('t')

# (e) Невязка и её производная
axes[2, 0].plot(nu_ad, 'k', lw=0.5, label='ν(t)')
axes[2, 0].plot(nu_dot_ad * 5, 'r', lw=0.5, alpha=0.7, label='ν̇(t) × 5')
axes[2, 0].set_title('(e) Невязка и её производная')
axes[2, 0].set_xlabel('t')
axes[2, 0].legend(fontsize=8)

# (f) Ошибки
err_cl = np.abs(x_true - x_cl)
err_ad = np.abs(x_true - x_ad)
axes[2, 1].plot(err_cl, 'b', lw=0.5, alpha=0.7, label='классический')
axes[2, 1].plot(err_ad, 'r', lw=0.5, label='PD (адапт. D)')
axes[2, 1].set_title('(f) Ошибка |x_true - x̂|')
axes[2, 1].set_xlabel('t')
axes[2, 1].legend(fontsize=8)

for ax in axes.flat:
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, N)

plt.suptitle('PD-Калман: производная невязки в фильтре',
             fontsize=16, y=1.00)
plt.tight_layout()

os.makedirs('results', exist_ok=True)
fig.savefig('results/pd_kalman.png', dpi=200, bbox_inches='tight')
print("\nГрафик сохранён: results/pd_kalman.png")
plt.show()