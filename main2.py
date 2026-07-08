import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import shutil
from google.colab import files

# ==========================================
# 1. 共通ベースラインパラメータの設定
# ==========================================
# Table 20: 第三者介入を伴うシミュレーションパラメータに準拠
N = 71              # コミュニティサイズ [cite: 42]
sigma_s = 0.89      # 社会的影響力 [cite: 42]
rho = 0.74          # 非協力度 [cite: 42]
kappa = 0.52        # 攻撃係数 [cite: 42]
lam = 0.83          # ダメージ増加率 [cite: 42]
p = 0.01            # 情報伝播係数 [cite: 42]
u1_strength = 0.41  # 介入強度 [cite: 43]
theta = 1.0         # 利得の歪み係数（基準値）

# ==========================================
# 2. 物理演算と社会行動の連成力学モデル
# ==========================================
def coupled_dynamics(y, t, mu_val, t_int):
    I_s, D_s, E_eff, I_health = y

    # 状態変数の境界制約
    I_s = np.clip(I_s, 0.0, 1.0)
    I_health = np.clip(I_health, 0.0, 1.0)

    # 介入の有無 (指定されたタイミング t_int で発動)
    u_1 = u1_strength if t >= t_int else 0.0

    # 期待利得と複雑な伝染による拡散乗数（物理計算結果の統合）
    pi_t = 1.0 - (1.0 - I_s)**theta
    f_pi = 1.0 + 3.0 * pi_t
    gamma = 0.1 * I_health

    # 微分方程式系
    dIs_dt = p * (1 + u_1) * (1 - sigma_s) * (1 - rho) * (N - 1) * (1 - I_s) * f_pi - gamma * I_s
    dDs_dt = lam * (1 - I_s) - mu_val * I_s + kappa * rho * (1 - I_s) * sigma_s

    # 相転移トリガー (D_s が負に突入した際に E_eff を爆発させる非線形フィードバック)
    phase_transition_multiplier = 0.0
    if D_s < 0 and t > 150:
        # 計算機上のオーバーフローを防ぐため、成長率を適度に制限
        phase_transition_multiplier = 0.5 * min(-D_s, 50.0) * E_eff

    dEeff_dt = 0.1 * (1 - E_eff) - 0.05 * max(0, D_s) * E_eff + 10.0 * pi_t + phase_transition_multiplier
    dIhealth_dt = 0.05 * (I_s - I_health) - 0.01 * (1 - gamma)

    return [dIs_dt, dDs_dt, dEeff_dt, dIhealth_dt]

# ==========================================
# 3. 不安定要素の計算実験実行
# ==========================================
output_dir = "unstable_elements_analysis"
os.makedirs(output_dir, exist_ok=True)

# 初期条件 [cite: 42]
y0 = [0.12, 79.28, 0.01, 0.15] # [I_s(0), D_s(0), E_eff(0), I_health(0)]
t = np.linspace(0, 200, 2000)

# 検証パラメータ空間
mu_variations = [0.14, 0.50, 1.00]      # 0.14(基準), 0.50, 1.00 (回復の非線形増幅を想定)
t_int_variations = [20.0, 59.0, 100.0]  # 59.0(基準), 早期介入(20), 遅延介入(100)

results = {}

# シミュレーションループ
for mu_val in mu_variations:
    for t_int in t_int_variations:
        sol = odeint(coupled_dynamics, y0, t, args=(mu_val, t_int))
        df = pd.DataFrame({
            'Time': t,
            'Is': sol[:, 0],
            'Ds': sol[:, 1],
            'Eeff': sol[:, 2],
            'Ihealth': sol[:, 3]
        })
        key = f"mu_{mu_val}_tint_{t_int}"
        results[key] = df
        df.to_csv(os.path.join(output_dir, f"sim_{key}.csv"), index=False)

# ==========================================
# 4. 結果の可視化と解析グラフ生成
# ==========================================
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Computational Experiment: Unstable Elements (Recovery Rate $\mu$ & Intervention Timing $t_{int}$)', fontsize=16)

colors = ['blue', 'red', 'green']
line_styles = ['-', '--', ':']

# プロット1: μの変化によるダメージ(Ds)の減衰軌道 (t_int=59固定)
ax1 = axes[0, 0]
for i, mu_val in enumerate(mu_variations):
    df = results[f"mu_{mu_val}_tint_59.0"]
    ax1.plot(df['Time'], df['Ds'], label=f"$\mu$={mu_val}", color=colors[i])
ax1.axvline(x=59.0, color='gray', linestyle='--', label='Intervention')
ax1.axhline(y=0, color='black', linestyle=':')
ax1.set_title('Damage Decay vs Recovery Rate ($\mu$)')
ax1.set_ylabel('Damage ($D_s$)')
ax1.legend()

# プロット2: μの変化による相転移時刻(t_c)のシフト (t_int=59固定)
ax2 = axes[0, 1]
for i, mu_val in enumerate(mu_variations):
    df = results[f"mu_{mu_val}_tint_59.0"]
    ax2.plot(df['Time'], df['Eeff'], label=f"$\mu$={mu_val}", color=colors[i])
ax2.axvline(x=59.0, color='gray', linestyle='--')
ax2.set_yscale('log')
ax2.set_title('Phase Transition Timing vs Recovery Rate ($\mu$)')
ax2.set_ylabel('Econ Efficiency ($E_{eff}$) [Log]')
ax2.legend()

# プロット3: 介入タイミング(t_int)の変化によるダメージ(Ds)軌道 (μ=0.50固定)
ax3 = axes[1, 0]
for i, t_int in enumerate(t_int_variations):
    df = results[f"mu_0.5_tint_{t_int}"]
    ax3.plot(df['Time'], df['Ds'], label=f"$t_{{int}}$={t_int}", color=colors[i], linestyle=line_styles[i])
ax3.axhline(y=0, color='black', linestyle=':')
ax3.set_title('Damage Trajectory vs Intervention Timing ($t_{int}$)')
ax3.set_xlabel('Time')
ax3.set_ylabel('Damage ($D_s$)')
ax3.legend()

# プロット4: 介入タイミング(t_int)の変化による相転移(Eeff) (μ=0.50固定)
ax4 = axes[1, 1]
for i, t_int in enumerate(t_int_variations):
    df = results[f"mu_0.5_tint_{t_int}"]
    ax4.plot(df['Time'], df['Eeff'], label=f"$t_{{int}}$={t_int}", color=colors[i], linestyle=line_styles[i])
ax4.set_yscale('log')
ax4.set_title('Phase Transition vs Intervention Timing ($t_{int}$)')
ax4.set_xlabel('Time')
ax4.set_ylabel('Econ Efficiency ($E_{eff}$) [Log]')
ax4.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join(output_dir, "unstable_elements_plots.png")
plt.savefig(plot_path, dpi=300)
plt.show()

# ==========================================
# 5. 結果のZIPアーカイブとダウンロード
# ==========================================
zip_filename = "unstable_elements_experiment"
shutil.make_archive(zip_filename, 'zip', output_dir)
print(f"計算実験が完了しました。アーカイブ '{zip_filename}.zip' をダウンロードします。")
files.download(f"{zip_filename}.zip")
