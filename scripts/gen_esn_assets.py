#!/usr/bin/env python3
import argparse
import numpy as np

from reservoirpy.datasets import mackey_glass, to_forecasting
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import rmse, rsquare

def scale_to_spectral_radius(W: np.ndarray, sr: float) -> np.ndarray:
    # units=100程度ならeigvalsでOK（PoC用途）
    eig = np.linalg.eigvals(W.astype(np.float64))
    radius = np.max(np.abs(eig))
    if radius == 0:
        raise ValueError("spectral radius is zero")
    return (W / radius * sr).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=2000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=100)

    ap.add_argument("--units", type=int, default=100)
    ap.add_argument("--sr", type=float, default=1.25)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=1e-5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- dataset ---
    X = mackey_glass(n_timesteps=args.timesteps).astype(np.float32)  # (T, 1)
    x_train, x_test, y_train, y_test = to_forecasting(X, test_size=args.test_size)
    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    Din = x_train.shape[1]
    Dout = y_train.shape[1]

    # --- make dense W/Win explicitly (rc-benchがdense前提のため) ---
    W_raw = rng.uniform(-0.5, 0.5, size=(args.units, args.units)).astype(np.float32)
    W = scale_to_spectral_radius(W_raw, args.sr)

    # ReservoirPyのWinは(Units, Features)として扱われることが多いので、この形で作る
    Win = rng.uniform(-0.5, 0.5, size=(args.units, Din)).astype(np.float32)

    bias = np.zeros((args.units,), dtype=np.float32)

    reservoir = Reservoir(units=args.units, lr=args.lr, W=W, Win=Win, bias=bias)
    readout = Ridge(ridge=args.ridge, fit_bias=True)
    esn = reservoir >> readout

    esn.fit(x_train, y_train, warmup=args.warmup)

    x0 = np.array(reservoir.state["out"], dtype=np.float32).reshape(1, args.units)  # (1, N)

    preds = esn.run(x_test).astype(np.float32)

    print(f"[assets] RMSE: {rmse(y_test, preds)}; R^2: {rsquare(y_test, preds)}")

    # readout parameters
    Wout = np.array(readout.Wout, dtype=np.float32)          # (N, Dout) 期待
    bout = np.array(readout.bias, dtype=np.float32).reshape(1, Dout)

    WinT = Win.T.copy()  # (Din, N)
    WT = W.T.copy().astype(np.float32)

    np.savez(
        args.out,
        Din=np.int64(Din), Dout=np.int64(Dout), N=np.int64(args.units),
        Ttest=np.int64(x_test.shape[0]),
        W=W, WT=WT, WinT=WinT,  # ★ WT を追加
        bias_res=bias, Wout=Wout, bias_out=bout, x0=x0,
        x_test=x_test, y_test=y_test,
        lr=np.float32(args.lr),
    )

    print(f"[assets] wrote: {args.out}")

if __name__ == "__main__":
    main()
