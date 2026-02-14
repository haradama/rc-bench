# scripts/verify_esn_ref.py（例）
import numpy as np

z = np.load("results_phase2/esn_mg_assets.npz")

W     = z["W"].astype(np.float32)              # (N,N)  ※列ベクトル前提
WinT  = z["WinT"].astype(np.float32)           # (Din,N)
Win   = WinT.T                                 # (N,Din)
Wout  = z["Wout"].astype(np.float32)           # (N,Dout)
bout  = z["bias_out"].astype(np.float32).reshape(-1)  # (Dout,)
biasr = z.get("bias_res", np.zeros((W.shape[0],), np.float32)).astype(np.float32)

X = z["x_test"].astype(np.float32)             # (Ttest,Din)
Y = z["y_test"].astype(np.float32)             # (Ttest,Dout)
x = z["x0"].astype(np.float32).reshape(-1)     # (N,)
lr = float(z["lr"])

pred = np.empty_like(Y)
one_minus = np.float32(1.0 - lr)

for t in range(X.shape[0]):
    u = X[t]                                   # (Din,)
    pre = W @ x + Win @ u + biasr              # (N,)
    x = one_minus * x + np.float32(lr) * np.tanh(pre).astype(np.float32)
    pred[t] = (x @ Wout + bout).astype(np.float32)

rmse = float(np.sqrt(np.mean((pred - Y) ** 2)))
ss_res = float(np.sum((Y - pred) ** 2))
ss_tot = float(np.sum((Y - Y.mean()) ** 2))
r2 = 0.0 if ss_tot == 0.0 else (1.0 - ss_res / ss_tot)

print("ref_rmse=", rmse)
print("ref_r2=", r2)
