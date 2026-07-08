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
N = 100             # コミュニティサイズ
m = 2               # スケールフリーネットワークの新規エッジ数
sigma_s = 0.89      # 社会的影響力
rho = 0.74          # 非協力度
kappa = 0.52        # 攻撃係数
lam = 0.83          # ダメージ増加率
mu_base = 0.14      # 基礎ダメージ回復率
p = 0.01            # 基礎伝播係数
theta = 1.0         # 利得の歪み係数

t_intervention = 59.0
intervention_fraction = 0.10  # 上位10%のハブノードを初期介入対象とする

# ==========================================
# 2. 適応的ネットワーク・シミュレーション関数
# ==========================================
def run_adaptive_simulation(strategy_name, use_dynamic_mu=False, use_rewiring=False, alpha_mu=0.02, rewire_interval=10, damage_threshold=70.0):
    """
    動的回復力伝播と適応的再配線（Co-evolution）を統合したシミュレーション
    """
    # ネットワークの初期化 (シナリオごとに独立した初期状態を担保)
    G = nx.barabasi_albert_graph(N, m, seed=42)
    A = nx.to_numpy_array(G)
    
    # ハブの特定
    degrees = np.array([G.degree(n) for n in range(N)])
    hub_indices = np.argsort(degrees)[-int(N * intervention_fraction):]

    dt = 0.1
    t_steps = int(200 / dt)
    time_array = np.linspace(0, 200, t_steps)
    
    # 状態変数の初期化（局所ベクトル化）
    I = np.full(N, 0.12)
    D = np.full(N, 79.28)
    I_health = np.full(N, 0.15) # ノードごとの情報健全性
    mu = np.full(N, mu_base)
    
    E_eff = 0.01 # マクロ経済効率性
    
    # 記録用リスト
    records = { 'Time': [], 'Is_mean': [], 'Ds_mean': [], 'Eeff': [], 'Edge_Density': [] }
    
    p_scale = p * 15.0 
    
    for step, t in enumerate(time_array):
        # --------------------------------------------------
        # A. 初期介入（ハブへのファクトチェック/権限付与）
        # --------------------------------------------------
        if t >= t_intervention and t < t_intervention + dt:
            mu[hub_indices] = 1.0  # ハブの回復力を強制ブースト
            I_health[hub_indices] = 0.9 # ハブの健全性を担保

        # --------------------------------------------------
        # B. 適応的再配線 (Adaptive Rewiring) の力学
        # --------------------------------------------------
        if use_rewiring and t >= t_intervention and step % rewire_interval == 0:
            for i in range(N):
                if D[i] > damage_threshold: # ダメージが閾値を超えたエージェントが自律行動
                    neighbors = np.where(A[i] > 0)[0]
                    if len(neighbors) > 0:
                        # 感染度(I)が最も高い汚染源ノードを特定
                        worst_neighbor = neighbors[np.argmax(I[neighbors])]
                        
                        # 汚染源の感染度が顕著(0.5以上)であればリンク切断
                        if I[worst_neighbor] > 0.5:
                            A[i, worst_neighbor] = 0
                            A[worst_neighbor, i] = 0
                            
                            # ネットワーク全体から新たな健全ノード（ゲートキーパー）を探索
                            non_neighbors = np.where(A[i] == 0)[0]
                            non_neighbors = non_neighbors[non_neighbors != i]
                            if len(non_neighbors) > 0:
                                # 健全性(I_health)が高く、感染度(I)が低いノードを優先
                                attractiveness = I_health[non_neighbors] - I[non_neighbors]
                                best_candidate = non_neighbors[np.argmax(attractiveness)]
                                # 再配線
                                A[i, best_candidate] = 1
                                A[best_candidate, i] = 1

        # --------------------------------------------------
        # C. 微分方程式系 (Euler法による更新)
        # --------------------------------------------------
        # 利得と拡散乗数
        pi_t = 1.0 - (1.0 - I)**theta
        f_pi = 1.0 + 3.0 * pi_t
        
        # 隣接行列を用いた局所相互作用
        neighbor_infection = A.dot(I * f_pi)
        gamma = 0.1 * I_health
        
        # 動的パラメータの微分 (回復力 mu の伝染)
        if use_dynamic_mu and t >= t_intervention:
            # 健全な隣接ノードとの接触により自己の回復力が同調（上昇）する
            dmu_dt = alpha_mu * A.dot(I_health) - 0.05 * (mu - mu_base)
        else:
            dmu_dt = np.zeros(N)

        # 状態変数の微分
        dI_dt = p_scale * (1 - I) * neighbor_infection - gamma * I
        dD_dt = lam * (1 - I) - mu * I + kappa * rho * (1 - I) * sigma_s
        dIhealth_dt = 0.05 * (1.0 - I) - 0.02 * I_health # 感染度が低いほど健全性が回復
        
        # マクロ指標と相転移トリガー
        D_mean = np.mean(D)
        phase_transition_multiplier = 0.0
        if D_mean < 0 and t > 150:
            phase_transition_multiplier = 0.5 * min(-D_mean, 50.0) * E_eff
            
        dEeff_dt = 0.1 * (1 - E_eff) - 0.05 * max(0, D_mean) * E_eff + 10.0 * np.mean(pi_t) + phase_transition_multiplier
        
        # 状態の更新とクリッピング
        I = np.clip(I + dI_dt * dt, 0.0, 1.0)
        D = D + dD_dt * dt
        mu = np.clip(mu + dmu_dt * dt, mu_base, 2.0)
        I_health = np.clip(I_health + dIhealth_dt * dt, 0.0, 1.0)
        E_eff = E_eff + dEeff_dt * dt
        
        # 記録
        records['Time'].append(t)
        records['Is_mean'].append(np.mean(I))
        records['Ds_mean'].append(np.mean(D))
        records['Eeff'].append(E_eff)
        records['Edge_Density'].append(np.sum(A) / (N*(N-1)))

    return pd.DataFrame(records)

# ==========================================
# 3. 仮説検証のための4象限シナリオ実行
# ==========================================
output_dir = "adaptive_network_experiment"
os.makedirs(output_dir, exist_ok=True)

# 不安定パラメータを網羅的に比較する実験セット
scenarios = {
    '1. Static Hub (Baseline)': {'use_dynamic_mu': False, 'use_rewiring': False},
    '2. Dynamic Hygiene Only':  {'use_dynamic_mu': True,  'use_rewiring': False},
    '3. Adaptive Rewiring Only':{'use_dynamic_mu': False, 'use_rewiring': True},
    '4. Full Co-evolution':     {'use_dynamic_mu': True,  'use_rewiring': True}
}

results = {}
for name, params in scenarios.items():
    print(f"Running simulation: {name}...")
    df = run_adaptive_simulation(name, **params)
    results[name] = df
    df.to_csv(os.path.join(output_dir, f"{name.replace(' ', '_')}.csv"), index=False)

# ==========================================
# 4. グラフ化と相転移の可視化
# ==========================================
plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Adaptive Network Co-evolution & Dynamic Hygiene Contagion', fontsize=16)

colors = ['gray', 'blue', 'orange', 'green']
line_styles = [':', '--', '-.', '-']

# 1. 感染度 (Is)
for i, (name, df) in enumerate(results.items()):
    axes[0].plot(df['Time'], df['Is_mean'], label=name, color=colors[i], linestyle=line_styles[i], linewidth=2)
axes[0].axvline(x=t_intervention, color='red', linestyle='--', alpha=0.5, label='Intervention')
axes[0].set_title('Mean Infection ($I_s$)')
axes[0].set_ylabel('Ratio of Infection')
axes[0].legend(loc='lower right', fontsize=9)

# 2. ダメージ (Ds)
for i, (name, df) in enumerate(results.items()):
    axes[1].plot(df['Time'], df['Ds_mean'], label=name, color=colors[i], linestyle=line_styles[i], linewidth=2)
axes[1].axvline(x=t_intervention, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(y=0, color='black', linestyle='-')
axes[1].set_title('Mean Damage ($D_s$)')
axes[1].set_ylabel('Damage')

# 3. 経済効率性 (Eeff) - Log Scale
for i, (name, df) in enumerate(results.items()):
    axes[2].plot(df['Time'], df['Eeff'], label=name, color=colors[i], linestyle=line_styles[i], linewidth=2)
axes[2].axvline(x=t_intervention, color='red', linestyle='--', alpha=0.5)
axes[2].set_yscale('log')
axes[2].set_title('Econ Efficiency ($E_{eff}$) Phase Transition')
axes[2].set_ylabel('Eeff (Log Scale)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join(output_dir, "adaptive_network_comparison.png")
plt.savefig(plot_path, dpi=300)
plt.show()

# ==========================================
# 5. 結果のZIPパッケージ化
# ==========================================
zip_filename = "adaptive_network_experiment_results"
shutil.make_archive(zip_filename, 'zip', output_dir)
print(f"\n実験が完了しました。'{zip_filename}.zip' をダウンロードします。")
files.download(f"{zip_filename}.zip")
