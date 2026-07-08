import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import shutil
from google.colab import files

# ==========================================
# 1. ネットワークと共通パラメータの設定
# ==========================================
N = 100             # コミュニティサイズ（ネットワーク効果を可視化するため100に設定）
m = 2               # スケールフリーネットワークの新規エッジ数
sigma_s = 0.89      # 社会的影響力
rho = 0.74          # 非協力度
kappa = 0.52        # 攻撃係数
lam = 0.83          # ダメージ増加率
mu_base = 0.14      # 基礎ダメージ回復率
mu_boost = 1.00     # 介入による構造的回復力の増幅値
p = 0.01            # 基礎伝播係数
theta = 1.0         # 利得の歪み係数

t_intervention = 59.0
intervention_fraction = 0.10  # 全体の10%のノードに介入

# スケールフリーネットワーク（Barabasi-Albert）の生成
G = nx.barabasi_albert_graph(N, m, seed=42)
A = nx.to_numpy_array(G)  # 隣接行列

# ノードの次数（つながり数）を取得し、ハブノードを特定
degrees = np.array([G.degree(n) for n in range(N)])
hub_indices = np.argsort(degrees)[-int(N * intervention_fraction):] # 上位10%のハブ
random_indices = np.random.choice(N, int(N * intervention_fraction), replace=False)

# ==========================================
# 2. ネットワーク結合型シミュレーション関数
# ==========================================
def run_network_simulation(strategy_name, intervention_nodes):
    dt = 0.1
    t_steps = int(200 / dt)
    time_array = np.linspace(0, 200, t_steps)
    
    # 状態変数の初期化 (各ノードごとの配列)
    I = np.full(N, 0.12)
    D = np.full(N, 79.28)
    E_eff = 0.01 # 経済効率性は社会全体（マクロ）の指標として扱う
    I_health = 0.15 # 情報健全性もマクロ指標とする
    
    # ノードごとの回復力配列
    mu_array = np.full(N, mu_base)
    
    # 記録用リスト
    record_Is_mean = []
    record_Ds_mean = []
    record_Eeff = []
    
    # ネットワーク構造における拡散のスケール調整（平均場との整合性）
    p_scale = p * 15.0 
    
    for t in time_array:
        # 介入タイミングでの構造的変化（対象ノードの回復力増幅）
        if t >= t_intervention and strategy_name != 'No Intervention':
            mu_array[intervention_nodes] = mu_boost
            
        # 期待利得と拡散乗数（ノードごと）
        pi_t = 1.0 - (1.0 - I)**theta
        f_pi = 1.0 + 3.0 * pi_t
        
        # 隣接ノードからの情報伝播の影響（隣接行列による高速計算）
        neighbor_influence = A.dot(I * f_pi)
        gamma = 0.1 * I_health
        
        # 微分（差分）の計算
        dI_dt = p_scale * (1 - I) * neighbor_influence - gamma * I
        dD_dt = lam * (1 - I) - mu_array * I + kappa * rho * (1 - I) * sigma_s
        
        # マクロ指標の計算
        D_mean = np.mean(D)
        I_mean = np.mean(I)
        
        # 相転移トリガー
        phase_transition_multiplier = 0.0
        if D_mean < 0 and t > 150:
            phase_transition_multiplier = 0.5 * min(-D_mean, 50.0) * E_eff
            
        dEeff_dt = 0.1 * (1 - E_eff) - 0.05 * max(0, D_mean) * E_eff + 10.0 * np.mean(pi_t) + phase_transition_multiplier
        dIhealth_dt = 0.05 * (I_mean - I_health) - 0.01 * (1 - gamma)
        
        # 状態の更新 (オイラー法)
        I = np.clip(I + dI_dt * dt, 0.0, 1.0)
        D = D + dD_dt * dt
        E_eff = E_eff + dEeff_dt * dt
        I_health = np.clip(I_health + dIhealth_dt * dt, 0.0, 1.0)
        
        # 記録
        record_Is_mean.append(np.mean(I))
        record_Ds_mean.append(np.mean(D))
        record_Eeff.append(E_eff)
        
    return pd.DataFrame({
        'Time': time_array,
        'Is_mean': record_Is_mean,
        'Ds_mean': record_Ds_mean,
        'Eeff': record_Eeff
    })

# ==========================================
# 3. 実験パターンの実行
# ==========================================
output_dir = "network_intervention_analysis"
os.makedirs(output_dir, exist_ok=True)

strategies = {
    'No Intervention': [],
    'Random Intervention (10%)': random_indices,
    'Targeted Intervention (Hub 10%)': hub_indices
}

results = {}
for name, nodes in strategies.items():
    print(f"Running simulation: {name}...")
    df = run_network_simulation(name, nodes)
    results[name] = df
    df.to_csv(os.path.join(output_dir, f"{name.replace(' ', '_')}.csv"), index=False)

# ==========================================
# 4. 可視化とプロット
# ==========================================
plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Targeted vs Random Intervention on Scale-Free Network', fontsize=16)

colors = ['red', 'blue', 'green']
line_styles = [':', '--', '-']

# 1. Information State (Is)
for i, (name, df) in enumerate(results.items()):
    axes[0].plot(df['Time'], df['Is_mean'], label=name, color=colors[i], linestyle=line_styles[i])
axes[0].axvline(x=t_intervention, color='gray', linestyle='--', alpha=0.7)
axes[0].set_title('Mean Info State ($I_s$)')
axes[0].set_ylabel('Ratio of Infection')
axes[0].legend()

# 2. Damage (Ds)
for i, (name, df) in enumerate(results.items()):
    axes[1].plot(df['Time'], df['Ds_mean'], label=name, color=colors[i], linestyle=line_styles[i])
axes[1].axvline(x=t_intervention, color='gray', linestyle='--', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-')
axes[1].set_title('Mean Damage ($D_s$) Decay')
axes[1].set_ylabel('Damage')
axes[1].legend()

# 3. Economic Efficiency (Eeff) - Log Scale
for i, (name, df) in enumerate(results.items()):
    axes[2].plot(df['Time'], df['Eeff'], label=name, color=colors[i], linestyle=line_styles[i])
axes[2].axvline(x=t_intervention, color='gray', linestyle='--', alpha=0.7)
axes[2].set_yscale('log')
axes[2].set_title('Econ Efficiency ($E_{eff}$) Phase Transition')
axes[2].set_ylabel('Eeff (Log)')
axes[2].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join(output_dir, "network_intervention_comparison.png")
plt.savefig(plot_path, dpi=300)
plt.show()

# ==========================================
# 5. ZIP化してダウンロード
# ==========================================
zip_filename = "network_intervention_experiment"
shutil.make_archive(zip_filename, 'zip', output_dir)
print(f"\n実験が完了しました。'{zip_filename}.zip' をダウンロードします。")
files.download(f"{zip_filename}.zip")
