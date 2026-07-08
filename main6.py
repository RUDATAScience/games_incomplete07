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
m = 2               # Barabasi-Albertモデルの新規エッジ数
sigma_s = 0.89      # 社会的影響力
rho = 0.74          # 非協力度
kappa = 0.52        # 攻撃係数
lam = 0.83          # ダメージ増加率
mu_base = 0.14      # 基礎ダメージ回復率
p = 0.01            # 基礎伝播係数
theta = 1.0         # 利得の歪み係数

# ==========================================
# 2. 共進化・適応的ネットワークシミュレーション関数
# ==========================================
def run_adaptive_coevolution(strategy_name, use_edge_deletion=False, use_circuit_breaker=False, 
                             damage_threshold=70.0, cb_trigger_Is=0.4, cb_p_penalty=0.1):
    """
    動的エッジ切断とサーキット・ブレーカー（大域的介入）を統合したシミュレーション
    """
    # ネットワークの初期化 (各シナリオで共通の初期構造)
    G = nx.barabasi_albert_graph(N, m, seed=42)
    A = nx.to_numpy_array(G)
    
    dt = 0.1
    t_steps = int(200 / dt)
    time_array = np.linspace(0, 200, t_steps)
    
    # 状態変数の初期化
    I = np.full(N, 0.12)
    D = np.full(N, 79.28)
    E_eff = 0.01 
    
    # スケールフリー用の伝播係数補正
    p_scale_base = p * 15.0 
    
    records = { 'Time': [], 'Is_mean': [], 'Ds_mean': [], 'Eeff': [], 'Edge_Density': [] }
    
    for step, t in enumerate(time_array):
        
        # --------------------------------------------------
        # A. サーキット・ブレーカー（大域的・アルゴリズム的介入）
        # --------------------------------------------------
        I_mean = np.mean(I)
        p_current = p_scale_base
        
        # ネットワーク全体の感染度が閾値を超えた場合、伝播係数を強制減衰
        if use_circuit_breaker and I_mean > cb_trigger_Is:
            p_current = p_scale_base * cb_p_penalty

        # --------------------------------------------------
        # B. 構造的バイパス（局所的・動的エッジ切断）
        # --------------------------------------------------
        # 計算負荷軽減と物理的妥当性（人間の反応遅延）のため、一定ステップごとに評価
        if use_edge_deletion and step % 10 == 0:
            for i in range(N):
                # エージェントの局所ダメージが限界を突破した場合
                if D[i] > damage_threshold:
                    neighbors = np.where(A[i] > 0)[0]
                    if len(neighbors) > 0:
                        # 自身に繋がる最も感染度が高いノード（最大の汚染源）を特定
                        worst_neighbor = neighbors[np.argmax(I[neighbors])]
                        
                        # 汚染源の感染度が顕著（0.5以上）であればブロック（リンク切断）
                        if I[worst_neighbor] > 0.5:
                            A[i, worst_neighbor] = 0
                            A[worst_neighbor, i] = 0

        # --------------------------------------------------
        # C. 物理演算（Euler法による連成方程式の更新）
        # --------------------------------------------------
        # 利得の歪みと複雑な伝染の乗数
        pi_t = 1.0 - (1.0 - I)**theta
        f_pi = 1.0 + 3.0 * pi_t
        
        # 隣接行列Aを用いた伝播演算
        neighbor_infection = A.dot(I * f_pi)
        gamma = np.full(N, 0.015) # 基礎的な抑制力
        
        # 微分計算
        dI_dt = p_current * (1 - I) * neighbor_infection - gamma * I
        dD_dt = lam * (1 - I) - mu_base * I + kappa * rho * (1 - I) * sigma_s
        
        D_mean = np.mean(D)
        
        # 臨界現象（相転移）のトリガー
        phase_transition_multiplier = 0.0
        if D_mean < 0 and t > 150:
            phase_transition_multiplier = 0.5 * min(-D_mean, 50.0) * E_eff
            
        dEeff_dt = 0.1 * (1 - E_eff) - 0.05 * max(0, D_mean) * E_eff + 10.0 * np.mean(pi_t) + phase_transition_multiplier
        
        # 変数の更新
        I = np.clip(I + dI_dt * dt, 0.0, 1.0)
        D = D + dD_dt * dt
        E_eff = E_eff + dEeff_dt * dt
        
        # データ記録
        records['Time'].append(t)
        records['Is_mean'].append(I_mean)
        records['Ds_mean'].append(D_mean)
        records['Eeff'].append(E_eff)
        records['Edge_Density'].append(np.sum(A) / (N * (N - 1))) # ネットワーク密度の推移

    return pd.DataFrame(records)

# ==========================================
# 3. 網羅的検証シナリオの実行
# ==========================================
output_dir = "adaptive_network_hypothesis"
os.makedirs(output_dir, exist_ok=True)

# 4つの象限で仮説の有効性をテスト
scenarios = {
    '1. Static Network (Baseline)': {'use_edge_deletion': False, 'use_circuit_breaker': False},
    '2. Structural Bypass Only':    {'use_edge_deletion': True,  'use_circuit_breaker': False},
    '3. Circuit Breaker Only':      {'use_edge_deletion': False, 'use_circuit_breaker': True},
    '4. Full Co-evolution':         {'use_edge_deletion': True,  'use_circuit_breaker': True}
}

results = {}
for name, params in scenarios.items():
    print(f"Running simulation: {name}...")
    df = run_adaptive_coevolution(name, **params)
    results[name] = df
    df.to_csv(os.path.join(output_dir, f"{name.replace(' ', '_')}.csv"), index=False)

# ==========================================
# 4. 力学ダイナミクスの可視化
# ==========================================
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Adaptive Network Intervention: Structural Bypass vs Circuit Breaker', fontsize=16)

colors = ['gray', 'blue', 'orange', 'green']
line_styles = [':', '--', '-.', '-']

# (1) Information State (Is)
for i, (name, df) in enumerate(results.items()):
    axes[0, 0].plot(df['Time'], df['Is_mean'], label=name, color=colors[i], linestyle=line_styles[i], linewidth=2)
axes[0, 0].set_title('Mean Infection ($I_s$)')
axes[0, 0].set_ylabel('Ratio of Infection')
axes[0, 0].legend(loc='upper left', fontsize=9)

# (2) Damage (Ds)
for i, (name, df) in enumerate(results.items()):
    axes[0, 1].plot(df['Time'], df['Ds_mean'], label=name, color=colors[i], linestyle=line_styles[i], linewidth=2)
axes[0, 1].axhline(y=0, color='black', linestyle='-')
axes[0, 1].set_title('Mean Damage ($D_s$)')
axes[0, 1].set_ylabel('Damage')

# (3) Economic Efficiency (Eeff)
for i, (name, df) in enumerate(results.items()):
    axes[1, 0].plot(df['Time'], df['Eeff'], label=name, color=colors[i], linestyle=line_styles[i], linewidth=2)
axes[1, 0].set_yscale('log')
axes[1, 0].set_title('Econ Efficiency ($E_{eff}$) Phase Transition')
axes[1, 0].set_ylabel('Eeff (Log Scale)')
axes[1, 0].set_xlabel('Time')

# (4) Edge Density (ネットワーク構造の変化)
for i, (name, df) in enumerate(results.items()):
    axes[1, 1].plot(df['Time'], df['Edge_Density'], label=name, color=colors[i], linestyle=line_styles[i], linewidth=2)
axes[1, 1].set_title('Network Edge Density (Topology Decay)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_xlabel('Time')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join(output_dir, "adaptive_network_plots.png")
plt.savefig(plot_path, dpi=300)
plt.show()

# ==========================================
# 5. ZIP化してダウンロード
# ==========================================
zip_filename = "adaptive_network_simulation_results"
shutil.make_archive(zip_filename, 'zip', output_dir)
print(f"\n実験が完了しました。'{zip_filename}.zip' をダウンロードします。")
files.download(f"{zip_filename}.zip")
