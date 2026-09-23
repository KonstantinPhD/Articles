# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 19:02:14 2026

@author: assis
"""

# -*- coding: utf-8 -*-
"""
IAEKF с АДАПТИВНЫМ ОКНОМ через KLD.

Идея:
    На каждом шаге t выбираем размер окна w_t из набора кандидатов,
    минимизируя KLD между эмпирическим распределением невязки
    и теоретическим N(0, P^- + R_hat_w).

    w_t* = argmin_w D_KL( p_hat_w(nu) || N(0, P^- + R_hat_w) )

Сравнение:
    - фиксированное окно w = 50
    - адаптивное окно w_t ∈ {10, 20, 50, 100, 200}

Тест:
    - стационарный AR(1)
    - скачкообразное изменение R
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. ПАРАМЕТРЫ МОДЕЛИ
# ============================================================
np.random.seed(42)

N = 2000
a = 0.95
Q = 1.0

# R меняется скачком — тест на нестационарность
R_segments = [
    (0,    500,  1.0),   # R = 1
    (500,  1000, 4.0),   # R = 4 (скачок)
    (1000, 1500, 0.5),   # R = 0.5 (падение)
    (1500, N,    4.0),   # R = 4 (возврат)
]

w_fixed  = 50                        # фиксированное окно
w_candidates = [10, 20, 50, 100, 200]  # кандидаты для адаптивного окна
eps = 1e-6

# ============================================================
# 2. ГЕНЕРАЦИЯ ДАННЫХ С ПЕРЕМЕННОЙ R
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
# 3. ФУНКЦИЯ KLD ДЛЯ ГАУССОВЫХ РАСПРЕДЕЛЕНИЙ
# ============================================================
def kld_gaussian(var_emp, var_theor):
    """
    KLD между N(0, var_emp) и N(0, var_theor).
    J = 0.5 * [ var_emp/var_theor + log(var_theor) - log(var_emp) - 1 ]
    """
    if var_emp <= 0 or var_theor <= 0:
        return np.inf
    return 0.5 * (var_emp / var_theor + np.log(var_theor)
                  - np.log(var_emp) - 1.0)

# ============================================================
# 4. IAEKF С ФИКСИРОВАННЫМ ОКНОМ
# ============================================================
x_fixed = np.zeros(N)
P_fixed = np.zeros(N)
K_fixed = np.zeros(N)
R_fixed = np.zeros(N)
nu_fixed = np.zeros(N)

x_fixed[0] = y_obs[0]
P_fixed[0] = R_true[0]
R_fixed[0] = R_true[0]

for t in range(1, N):
    # Predict
    x_pred = a * x_fixed[t-1]
    P_pred = a**2 * P_fixed[t-1] + Q

    # Innovation
    nu_fixed[t] = y_obs[t] - x_pred

    # Empirical variance (fixed window)
    if t >= w_fixed:
        C_v = np.var(nu_fixed[t-w_fixed+1:t+1])
    else:
        C_v = P_pred + R_fixed[t-1]

    # Adaptive R
    R_fixed[t] = max(C_v - P_pred, eps)

    # Gain
    K_fixed[t] = P_pred / (P_pred + R_fixed[t])

    # Update
    x_fixed[t] = x_pred + K_fixed[t] * nu_fixed[t]
    P_fixed[t] = (1 - K_fixed[t]) * P_pred

# ============================================================
# 5. IAEKF С АДАПТИВНЫМ ОКНОМ
# ============================================================
x_adapt = np.zeros(N)
P_adapt = np.zeros(N)
K_adapt = np.zeros(N)
R_adapt = np.zeros(N)
nu_adapt = np.zeros(N)
w_adapt = np.zeros(N, dtype=int)
kld_adapt = np.zeros(N)

x_adapt[0] = y_obs[0]
P_adapt[0] = R_true[0]
R_adapt[0] = R_true[0]
w_adapt[0] = w_fixed

for t in range(1, N):
    # Predict
    x_pred = a * x_adapt[t-1]
    P_pred = a**2 * P_adapt[t-1] + Q

    # Innovation (не зависит от k_t!)
    nu_adapt[t] = y_obs[t] - x_pred

    # --------------------------------------------------------
    # ВЫБОР ОКНА через KLD
    # --------------------------------------------------------
    best_kld = np.inf
    best_w = w_fixed

    for w_c in w_candidates:
        if t < w_c:
            continue

        # Эмпирическая дисперсия невязки
        var_nu_w = np.var(nu_adapt[t-w_c+1:t+1])
        if var_nu_w <= 0:
            continue

        # Оценка R при данном окне
        R_w = max(var_nu_w - P_pred, eps)

        # Теоретическая дисперсия невязки
        var_theor = P_pred + R_w

        # KLD между эмпирическим и теоретическим
        J_w = kld_gaussian(var_nu_w, var_theor)

        if J_w < best_kld:
            best_kld = J_w
            best_w   = w_c

    w_adapt[t]   = best_w
    kld_adapt[t] = best_kld

    # --------------------------------------------------------
    # ОЦЕНКА R С ВЫБРАННЫМ ОКНОМ
    # --------------------------------------------------------
    if t >= best_w:
        C_v = np.var(nu_adapt[t-best_w+1:t+1])
    else:
        C_v = P_pred + R_adapt[t-1]

    R_adapt[t] = max(C_v - P_pred, eps)

    # Gain
    K_adapt[t] = P_pred / (P_pred + R_adapt[t])

    # Update
    x_adapt[t] = x_pred + K_adapt[t] * nu_adapt[t]
    P_adapt[t] = (1 - K_adapt[t]) * P_pred

# ============================================================
# 6. МЕТРИКИ
# ============================================================
rmse_fixed = np.sqrt(np.mean((x_true - x_fixed)**2))
rmse_adapt = np.sqrt(np.mean((x_true - x_adapt)**2))

# Ошибка оценки R
err_R_fixed = np.mean(np.abs(R_fixed - R_true))
err_R_adapt = np.mean(np.abs(R_adapt - R_true))

# Ошибка на границах скачков R (первые 50 шагов после скачка)
transition_mask = np.zeros(N, dtype=bool)
for (t0, t1, _) in R_segments[1:]:
    transition_mask[t0:t0+50] = True

err_R_fixed_trans = np.mean(np.abs(R_fixed[transition_mask]
                                   - R_true[transition_mask]))
err_R_adapt_trans = np.mean(np.abs(R_adapt[transition_mask]
                                   - R_true[transition_mask]))

print("=" * 60)
print("СРАВНЕНИЕ: ФИКСИРОВАННОЕ vs АДАПТИВНОЕ ОКНО")
print("=" * 60)
print(f"RMSE (фикс. окно w={w_fixed})      = {rmse_fixed:.4f}")
print(f"RMSE (адапт. окно)                 = {rmse_adapt:.4f}")
print(f"Средняя ошибка R (фикс.)           = {err_R_fixed:.4f}")
print(f"Средняя ошибка R (адапт.)          = {err_R_adapt:.4f}")
print(f"Ошибка R на скачках (фикс.)        = {err_R_fixed_trans:.4f}")
print(f"Ошибка R на скачках (адапт.)       = {err_R_adapt_trans:.4f}")
print(f"Улучшение на скачках               = "
      f"{(err_R_fixed_trans - err_R_adapt_trans)/err_R_fixed_trans*100:.1f}%")
print("=" * 60)

# Распределение выбранных окон
unique, counts = np.unique(w_adapt[100:], return_counts=True)
print("\nРаспределение выбранных окон:")
for u, c in zip(unique, counts):
    print(f"  w = {u:>4}: {c:>5} раз ({c/len(w_adapt[100:])*100:.1f}%)")
print("=" * 60)

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.25)

# (a) Истинный процесс и наблюдения
axes[0, 0].plot(x_true, 'k', lw=0.8, label='x_true')
axes[0, 0].plot(y_obs, 'gray', lw=0.4, alpha=0.5, label='y_obs')
axes[0, 0].set_title('(a) Истинный процесс и наблюдения')
axes[0, 0].legend(fontsize=8)

# (b) Оценки
axes[0, 1].plot(x_true, 'k', lw=0.8, label='x_true')
axes[0, 1].plot(x_fixed, 'b', lw=0.8, alpha=0.7, label='фикс. окно')
axes[0, 1].plot(x_adapt, 'r--', lw=0.8, label='адапт. окно')
axes[0, 1].set_title('(b) Оценки состояния')
axes[0, 1].legend(fontsize=8)

# (c) Коэффициент усиления K(t)
axes[1, 0].plot(K_fixed, 'b', lw=0.9, alpha=0.7, label='фикс. окно')
axes[1, 0].plot(K_adapt, 'r--', lw=0.9, label='адапт. окно')
axes[1, 0].set_title('(c) Коэффициент усиления K(t)')
axes[1, 0].legend(fontsize=8)

# (d) Оценка R
axes[1, 1].plot(R_true, 'k', lw=1.5, label='R_true')
axes[1, 1].plot(R_fixed, 'b', lw=0.8, alpha=0.7, label='фикс. окно')
axes[1, 1].plot(R_adapt, 'r--', lw=0.8, label='адапт. окно')
axes[1, 1].set_title('(d) Оценка R')
axes[1, 1].legend(fontsize=8)

# (e) Адаптивное окно
axes[2, 0].plot(w_adapt, 'r', lw=0.8)
axes[2, 0].axhline(y=w_fixed, color='b', ls='--', lw=1.5,
                   label=f'фикс. w = {w_fixed}')
axes[2, 0].set_title('(e) Адаптивное окно w_t')
axes[2, 0].set_ylim([0, max(w_candidates) + 20])
axes[2, 0].set_xlabel('t')
axes[2, 0].legend(fontsize=8)

# (f) KLD
axes[2, 1].plot(kld_adapt, 'purple', lw=0.6)
axes[2, 1].set_title('(f) KLD для выбранного окна')
axes[2, 1].set_xlabel('t')
axes[2, 1].set_yscale('log')

for ax in axes.flat:
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, N)

plt.suptitle('IAEKF: фиксированное vs адаптивное окно',
             fontsize=16, y=1.00)
plt.tight_layout()

os.makedirs('results', exist_ok=True)
fig.savefig('results/adaptive_window_kld.png', dpi=200, bbox_inches='tight')
print("\nГрафик сохранён: results/adaptive_window_kld.png")

plt.show()

# ============================================================
# 8. ТЕСТ НА СТАЦИОНАРНОМ R (для сравнения)
# ============================================================
print("\n" + "=" * 60)
print("ТЕСТ НА СТАЦИОНАРНОМ R")
print("=" * 60)

np.random.seed(42)
R_stat = 4.0
v_stat = np.random.normal(0, np.sqrt(R_stat), N)
y_stat = x_true + v_stat

def run_iaekf(y_obs, a, Q, window_fixed=None, adaptive=False):
    x = np.zeros(N)
    P = np.zeros(N)
    K = np.zeros(N)
    R_h = np.zeros(N)
    nu = np.zeros(N)
    w_arr = np.zeros(N, dtype=int)

    x[0] = y_obs[0]
    P[0] = R_stat
    R_h[0] = R_stat
    w_arr[0] = window_fixed if window_fixed else 50

    for t in range(1, N):
        x_pred = a * x[t-1]
        P_pred = a**2 * P[t-1] + Q
        nu[t] = y_obs[t] - x_pred

        if adaptive:
            best_kld = np.inf
            best_w = 50
            for w_c in w_candidates:
                if t < w_c: continue
                var_nu = np.var(nu[t-w_c+1:t+1])
                if var_nu <= 0: continue
                R_w = max(var_nu - P_pred, eps)
                J_w = kld_gaussian(var_nu, P_pred + R_w)
                if J_w < best_kld:
                    best_kld = J_w
                    best_w = w_c
            w_arr[t] = best_w
            C_v = np.var(nu[t-best_w+1:t+1])
        else:
            w_arr[t] = window_fixed
            if t >= window_fixed:
                C_v = np.var(nu[t-window_fixed+1:t+1])
            else:
                C_v = P_pred + R_h[t-1]

        R_h[t] = max(C_v - P_pred, eps)
        K[t] = P_pred / (P_pred + R_h[t])
        x[t] = x_pred + K[t] * nu[t]
        P[t] = (1 - K[t]) * P_pred

    return x, R_h, w_arr

x_f, R_f, w_f = run_iaekf(y_stat, a, Q, window_fixed=50, adaptive=False)
x_a, R_a, w_a = run_iaekf(y_stat, a, Q, adaptive=True)

rmse_f = np.sqrt(np.mean((x_true - x_f)**2))
rmse_a = np.sqrt(np.mean((x_true - x_a)**2))
err_R_f = np.mean(np.abs(R_f[-500:] - R_stat))
err_R_a = np.mean(np.abs(R_a[-500:] - R_stat))

print(f"R истинное                     = {R_stat:.4f}")
print(f"RMSE (фикс.)                   = {rmse_f:.4f}")
print(f"RMSE (адапт.)                  = {rmse_a:.4f}")
print(f"Ошибка R (фикс.)               = {err_R_f:.4f}")
print(f"Ошибка R (адапт.)              = {err_R_a:.4f}")
print(f"Среднее адаптивное окно        = {np.mean(w_a[-500:]):.1f}")
print("=" * 60)