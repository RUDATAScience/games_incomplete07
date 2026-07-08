import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import shutil
from google.colab import files

# ==========================================
# 1. 共通パラメータの設定
# ==========================================
N = 71              # コミュニティサイズ
sigma_s = 0.89      # 社会的影響力
rho = 0.74          # 非協力度
kappa = 0.52        # 攻撃係数
lam = 0.83          # ダメージ増加率
mu = 0.14           # ダメージ回復率

t_intervention = 59.0
u1_strength = 0.41

# ==========================================
# 2. 連成力学モデルの定義 (theta, p を引数化)
# ==========================================
def coupled_dynamics(y, t, theta, p_val):
    I_s, D_s, E_eff, I_health = y

    I_s = np.clip(I_s, 0.0, 1.0)
    I_health = np.clip(I_health, 0.0, 1.0)
    u_1 = u1_strength if t >= t_intervention else 0.0

    # 利得の非線形歪み (thetaによる制御)
    pi_t = 1.0 - (1.0 - I_s)**theta

    # 複雑な伝染 (最大4倍の拡散乗数)
    f_pi = 1.0 + 3.0 * pi_t
    gamma = 0.1 * I_health

    # 微分方程式
    dIs_dt = p_val * (1 + u_1) * (1 - sigma_s) * (1 - rho) * (N - 1) * (1 - I_s) * f_pi - gamma * I_s
    dDs_dt = lam * (1 - I_s) - mu * I_s + kappa * rho * (1 - I_s) * sigma_s

    # 相転移トリガー (Dsが負の領域で活性化)
    phase_transition_multiplier = 0.0
    if D_s < 0 and t > 150:
        phase_transition_multiplier = 0.5 * (-D_s) * E_eff

    dEeff_dt = 0.1 * (1 - E_eff) - 0.05 * max(0, D_s) * E_eff + 10.0 * pi_t + phase_transition_multiplier
    dIhealth_dt = 0.05 * (I_s - I_health) - 0.01 * (1 - gamma)

    return [dIs_dt, dDs_dt, dEeff_dt, dIhealth_dt]

# ==========================================
# 3. 感度分析の実行とデータ保存
# ==========================================
# 出力ディレクトリ作成
output_dir = "sensitivity_analysis_results"
os.makedirs(output_dir, exist_ok=True)

y0 = [0.12, 79.28, 0.01, 0.15]
t = np.linspace(0, 200, 2000)

# 検証するパラメータのリスト
theta_variations = [0.5, 1.0, 2.0]  # 0.5: 歪み大(悪化しやすい), 1.0: 基準, 2.0: 歪み小
p_variations = [0.005, 0.01, 0.02]  # 基準0.01に対する半減と倍増

results_dict = {}

# シミュレーションループ
for theta in theta_variations:
    for p_val in p_variations:
        sol = odeint(coupled_dynamics, y0, t, args=(theta, p_val))
        df = pd.DataFrame({
            'Time': t,
            'Is': sol[:, 0],
            'Ds': sol[:, 1],
            'Eeff': sol[:, 2],
            'Ihealth': sol[:, 3]
        })
        key = f"theta_{theta}_p_{p_val}"
        results_dict[key] = df

        # 個別CSVエクスポート
        df.to_csv(os.path.join(output_dir, f"data_{key}.csv"), index=False)

# ==========================================
# 4. 比較グラフの描画
# ==========================================
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Sensitivity Analysis: Non-linear Distortion (θ) & Propagation (p)', fontsize=18)

colors = ['blue', 'red', 'green']
line_styles = ['-', '--', ':']

# プロット1: Thetaの変化による経済効率性(Eeff)の比較 (p=0.01固定)
ax1 = axes[0, 0]
for i, theta in enumerate(theta_variations):
    df = results_dict[f"theta_{theta}_p_0.01"]
    ax1.plot(df['Time'], df['Eeff'], label=f"θ={theta}", color=colors[i])
ax1.axvline(x=t_intervention, color='gray', linestyle='--')
ax1.set_yscale('log')
ax1.set_title('E_eff Phase Transition vs Theta (p=0.01)')
ax1.set_ylabel('E_eff (Log Scale)')
ax1.legend()

# プロット2: Thetaの変化による情報状態(Is)の比較 (p=0.01固定)
ax2 = axes[0, 1]
for i, theta in enumerate(theta_variations):
    df = results_dict[f"theta_{theta}_p_0.01"]
    ax2.plot(df['Time'], df['Is'], label=f"θ={theta}", color=colors[i])
ax2.axvline(x=t_intervention, color='gray', linestyle='--')
ax2.set_title('Info Saturation (Is) vs Theta (p=0.01)')
ax2.set_ylabel('I_s')
ax2.legend()

# プロット3: pの変化による経済効率性(Eeff)の比較 (theta=1.0固定)
ax3 = axes[1, 0]
for i, p_val in enumerate(p_variations):
    df = results_dict[f"theta_1.0_p_{p_val}"]
    ax3.plot(df['Time'], df['Eeff'], label=f"p={p_val}", color=colors[i], linestyle=line_styles[i])
ax3.axvline(x=t_intervention, color='gray', linestyle='--')
ax3.set_yscale('log')
ax3.set_title('E_eff Phase Transition vs Prop. Coeff (p) (θ=1.0)')
ax3.set_xlabel('Time')
ax3.set_ylabel('E_eff (Log Scale)')
ax3.legend()

# プロット4: pの変化によるダメージ(Ds)の比較 (theta=1.0固定)
ax4 = axes[1, 1]
for i, p_val in enumerate(p_variations):
    df = results_dict[f"theta_1.0_p_{p_val}"]
    ax4.plot(df['Time'], df['Ds'], label=f"p={p_val}", color=colors[i], linestyle=line_styles[i])
ax4.axvline(x=t_intervention, color='gray', linestyle='--')
ax4.axhline(y=0, color='black', linestyle=':')
ax4.set_title('Damage (Ds) Recovery vs Prop. Coeff (p) (θ=1.0)')
ax4.set_xlabel('Time')
ax4.set_ylabel('D_s')
ax4.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, "sensitivity_comparison_plots.png"), dpi=300)
plt.show()

# ==========================================
# 5. ZIP圧縮と自動ダウンロード
# ==========================================
zip_filename = "sensitivity_analysis_archive"
shutil.make_archive(zip_filename, 'zip', output_dir)
print(f"解析が完了しました。'{zip_filename}.zip' をダウンロードします。")
files.download(f"{zip_filename}.zip")
