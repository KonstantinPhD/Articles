# -*- coding: utf-8 -*-
"""
PD-Калман для НЕЛИНЕЙНОГО тренда.

Сравнение:
    1. Классический Калман (f(t) = 0)
    2. PD-Калман (D_t из локальной γ)
    3. PID-Калман (D_t, E_t)
    4. Расширенный Калман (EKF)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================
np.random.seed(42)
N = 2000
a = 0.95
Q = 1.0
R = 4.0

# Нелинейный тренд
def f_trend(t, mode='quadratic'):
    if mode == 'quadratic':
        return 1e-6 * t**2
    elif mode == 'periodic':
        return 5 * np.sin(2 * np.pi * t / 500)
    elif mode == 'step':
        return 10 * (t > 1000)
    elif mode == 'exponential':
        return 0.01 * np.exp(0.002 * t)
    return 0.0

TREND_MODE = 'quadratic'
window = 50
D_max = 0.05

# ============================================================
# 2. ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================
w = np.random.normal(0, np.sqrt(Q), N)
v = np.random.normal(0, np.sqrt(R), N)

x_true = np.zeros(N)
x_true[0] = np.random.normal(0, np.sqrt(Q / (1 - a**2)))
for t in range(1, N):
    x_true[t] = a * x_true[t-1] + f_trend(t, TREND_MODE) + w[t]

y_obs = x_true + v

# ============================================================
# 3. КЛАССИЧЕСКИЙ КАЛМАН
# ============================================================
def run_classic(y, a, Q, R_init, window=50):
    N = len(y)
    x = np.zeros(N); P = np.zeros(N); K = np.zeros(N)
    nu = np.zeros(N); R_hat = np.zeros(N)
    x[0] = y[0]; P[0] = R_init; R_hat[0] = R_init
    for t in range(1, N):
        x_pred = a * x[t-1]
        P_pred = a**2 * P[t-1] + Q
        nu[t] = y[t] - x_pred
        if t >= window:
            C_v = np.var(nu[t-window+1:t+1])
        else:
            C_v = P_pred + R_hat[t-1]
        R_hat[t] = max(C_v - P_pred, 1e-6)
        K[t] = P_pred / (P_pred + R_hat[t])
        x[t] = x_pred + K[t] * nu[t]
        P[t] = (1 - K[t]) * P_pred
    return x, P, K, nu, R_hat

# ============================================================
# 4. PD-КАЛМАН с локальной γ
# ============================================================
def run_pd(y, a, Q, R_init, window=50, D_max=0.05):
    N = len(y)
    x = np.zeros(N); P = np.zeros(N); K = np.zeros(N)
    D_arr = np.zeros(N); nu = np.zeros(N); nu_dot = np.zeros(N)
    R_hat = np.zeros(N); gamma_arr = np.zeros(N)
    x[0] = y[0]; P[0] = R_init; R_hat[0] = R_init
    for t in range(1, N):
        x_pred = a * x[t-1]
        P_pred = a**2 * P[t-1] + Q
        nu[t] = y[t] - x_pred
        nu_dot[t] = nu[t] - nu[t-1] if t > 1 else 0.0

        if t >= window:
            C_v = np.var(nu[t-window+1:t+1])
        else:
            C_v = P_pred + R_hat[t-1]
        R_hat[t] = max(C_v - P_pred, 1e-6)
        K[t] = P_pred / (P_pred + R_hat[t])

        # Локальная γ (центрированная)
        if t >= window:
            nu_w = nu[t-window+1:t+1]
            nu_w_prev = nu[t-window:t]
            gamma_arr[t] = np.mean((nu_w - np.mean(nu_w))
                                   * (nu_w_prev - np.mean(nu_w_prev)))
        else:
            gamma_arr[t] = 0.0

        D_theory = P_pred * gamma_arr[t] / max(2 * (P_pred + R_hat[t])**2, 1e-6)
        D_arr[t] = np.clip(D_theory, 0.0, D_max)

        x[t] = x_pred + K[t] * nu[t] + D_arr[t] * nu_dot[t]
        P[t] = (1 - K[t]) * P_pred
    return x, P, K, D_arr, nu, nu_dot, R_hat, gamma_arr

# ============================================================
# 5. PID-КАЛМАН (вторая производная)
# ============================================================
def run_pid(y, a, Q, R_init, window=50, D_max=0.05, E_max=0.02):
    N = len(y)
    x = np.zeros(N); P = np.zeros(N); K = np.zeros(N)
    D_arr = np.zeros(N); E_arr = np.zeros(N)
    nu = np.zeros(N); nu_dot = np.zeros(N); nu_ddot = np.zeros(N)
    R_hat = np.zeros(N)
    x[0] = y[0]; P[0] = R_init; R_hat[0] = R_init
    for t in range(1, N):
        x_pred = a * x[t-1]
        P_pred = a**2 * P[t-1] + Q
        nu[t] = y[t] - x_pred
        nu_dot[t] = nu[t] - nu[t-1] if t > 1 else 0.0
        nu_ddot[t] = nu_dot[t] - nu_dot[t-1] if t > 2 else 0.0

        if t >= window:
            C_v = np.var(nu[t-window+1:t+1])
        else:
            C_v = P_pred + R_hat[t-1]
        R_hat[t] = max(C_v - P_pred, 1e-6)
        K[t] = P_pred / (P_pred + R_hat[t])

        if t >= window:
            nu_w = nu[t-window+1:t+1]
            nu_w_prev = nu[t-window:t]
            gamma = np.mean((nu_w - np.mean(nu_w))
                            * (nu_w_prev - np.mean(nu_w_prev)))
        else:
            gamma = 0.0

        D_arr[t] = np.clip(P_pred * gamma / max(2 * (P_pred + R_hat[t])**2, 1e-6),
                           0.0, D_max)
        E_arr[t] = 0.0  # упрощённо
        x[t] = (x_pred + K[t] * nu[t]
                + D_arr[t] * nu_dot[t]
                + E_arr[t] * nu_ddot[t])
        P[t] = (1 - K[t]) * P_pred
    return x, P, K, D_arr, E_arr, nu, nu_dot, nu_ddot, R_hat

# ============================================================
# 6. РАСШИРЕННЫЙ КАЛМАН (EKF) — квадратичный тренд
# ============================================================
def run_ekf_quadratic(y, a, Q, R, Q_c=1e-10):
    """
    Состояние: [x, c], f(t) = c * t^2.
    Модель: x_t = a*x_{t-1} + c_{t-1} * t^2 + w_t
            c_t = c_{t-1} + eta_t
    """
    N = len(y)
    z = np.zeros((N, 2))
    P = np.zeros((N, 2, 2))
    nu = np.zeros(N)
    z[0] = np.array([y[0], 0.0])
    P[0] = np.eye(2) * R

    for t in range(1, N):
        # Predict
        z_pred = np.array([a * z[t-1, 0] + z[t-1, 1] * t**2,
                           z[t-1, 1]])
        # Якобиан F_t
        F = np.array([[a, t**2],
                      [0.0, 1.0]])
        Q_mat = np.array([[Q, 0.0], [0.0, Q_c]])
        P_pred = F @ P[t-1] @ F.T + Q_mat

        # Innovation
        H = np.array([[1.0, 0.0]])
        nu[t] = y[t] - (H @ z_pred)[0]

        # Gain
        S = (H @ P_pred @ H.T + R)[0, 0]
        K = (P_pred @ H.T / S).flatten()

        # Update
        z[t] = z_pred + K * nu[t]
        P[t] = (np.eye(2) - np.outer(K, H)) @ P_pred

    return z, P, nu

# ============================================================
# 7. ЗАПУСК
# ============================================================
R_init = R

x_cl, _, _, nu_cl, _ = run_classic(y_obs, a, Q, R_init)
x_pd, _, _, D_pd, nu_pd, nu_dot_pd, _, gamma_pd = run_pd(y_obs, a, Q, R_init, D_max=D_max)
x_pid, _, _, D_pid, E_pid, nu_pid, nu_dot_pid, nu_ddot_pid, _ = run_pid(
    y_obs, a, Q, R_init, D_max=D_max, E_max=0.02)
z_ekf, _, nu_ekf = run_ekf_quadratic(y_obs, a, Q, R, Q_c=1e-10)
x_ekf = z_ekf[:, 0]
c_ekf = z_ekf[:, 1]

# ============================================================
# 8. МЕТРИКИ
# ============================================================
def rmse(x_true, x_hat):
    return np.sqrt(np.mean((x_true - x_hat)**2))

idx_ss = slice(N-500, N)

print("=" * 70)
print(f"НЕЛИНЕЙНЫЙ ТРЕНД: {TREND_MODE}")
print("=" * 70)
print(f"{'Метод':<35} {'RMSE':>10} {'RMSE(last500)':>15}")
print("-" * 70)
print(f"{'Классический':<35} {rmse(x_true, x_cl):>10.4f} "
      f"{rmse(x_true[idx_ss], x_cl[idx_ss]):>15.4f}")
print(f"{'PD-Калман (D_t)':<35} {rmse(x_true, x_pd):>10.4f} "
      f"{rmse(x_true[idx_ss], x_pd[idx_ss]):>15.4f}")
print(f"{'PID-Калман (D_t, E_t)':<35} {rmse(x_true, x_pid):>10.4f} "
      f"{rmse(x_true[idx_ss], x_pid[idx_ss]):>15.4f}")
print(f"{'EKF (оценка c)':<35} {rmse(x_true, x_ekf):>10.4f} "
      f"{rmse(x_true[idx_ss], x_ekf[idx_ss]):>15.4f}")
print("=" * 70)

# ============================================================
# 9. ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.25)

# (a) Истинный процесс
axes[0, 0].plot(x_true, 'k', lw=0.8, label='x_true')
axes[0, 0].plot(y_obs, 'gray', lw=0.3, alpha=0.4, label='y_obs')
axes[0, 0].set_title(f'(a) Истинный процесс ({TREND_MODE})')
axes[0, 0].legend(fontsize=8)

# (b) Оценки
axes[0, 1].plot(x_true, 'k', lw=0.8, label='x_true')
axes[0, 1].plot(x_cl, 'b', lw=0.6, alpha=0.7, label='классический')
axes[0, 1].plot(x_pd, 'orange', lw=0.6, alpha=0.8, label='PD')
axes[0, 1].plot(x_pid, 'g', lw=0.6, alpha=0.8, label='PID')
axes[0, 1].plot(x_ekf, 'r--', lw=0.7, label='EKF')
axes[0, 1].set_title('(b) Оценки')
axes[0, 1].legend(fontsize=7)

# (c) Ошибки
axes[1, 0].plot(x_true - x_cl, 'b', lw=0.5, alpha=0.7, label='классический')
axes[1, 0].plot(x_true - x_pd, 'orange', lw=0.5, alpha=0.8, label='PD')
axes[1, 0].plot(x_true - x_pid, 'g', lw=0.5, alpha=0.8, label='PID')
axes[1, 0].plot(x_true - x_ekf, 'r', lw=0.5, label='EKF')
axes[1, 0].axhline(y=0, color='k', ls=':', lw=0.5)
axes[1, 0].set_title('(c) Ошибка')
axes[1, 0].legend(fontsize=8)

# (d) D_t
axes[1, 1].plot(D_pd, 'orange', lw=0.9, label='D_t (PD)')
axes[1, 1].plot(D_pid, 'g', lw=0.9, label='D_t (PID)')
axes[1, 1].set_title('(d) Адаптивный D_t')
axes[1, 1].legend(fontsize=8)

# (e) Оценка c (EKF)
axes[2, 0].plot(c_ekf, 'r', lw=0.9, label='ĉ (EKF)')
axes[2, 0].axhline(y=1e-6, color='k', ls='--', lw=1.5, label='c_true')
axes[2, 0].set_title('(e) Оценка параметра тренда c')
axes[2, 0].legend(fontsize=8)

# (f) Невязка и производные
axes[2, 1].plot(nu_pd, 'k', lw=0.4, alpha=0.6, label='ν(t)')
axes[2, 1].plot(nu_dot_pd * 10, 'r', lw=0.4, alpha=0.6, label='ν̇(t)×10')
axes[2, 1].plot(nu_ddot_pid * 100, 'g', lw=0.4, alpha=0.6, label='ν̈(t)×100')
axes[2, 1].set_title('(f) Невязка и её производные')
axes[2, 1].legend(fontsize=8)

for ax in axes.flat:
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, N)

plt.suptitle(f'PD/PID/EKF для нелинейного тренда: {TREND_MODE}',
             fontsize=16, y=1.00)
plt.tight_layout()

import os
os.makedirs('results', exist_ok=True)
fig.savefig(f'results/pd_kalman_nonlinear_{TREND_MODE}.png',
            dpi=200, bbox_inches='tight')
print(f"\nГрафик сохранён: results/pd_kalman_nonlinear_{TREND_MODE}.png")
plt.show()