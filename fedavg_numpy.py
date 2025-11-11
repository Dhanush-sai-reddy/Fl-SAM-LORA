import os
import glob
import time
import json
import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np


# -----------------------------
# Utility helpers
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    o = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    o[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return o


def shuffle_data(X: np.ndarray, y: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = np.arange(X.shape[0])
        rng.shuffle(idx)
    else:
        idx = np.random.permutation(X.shape[0])
    return X[idx], y[idx]


# -----------------------------
# Model: Multiclass Logistic Regression (Softmax)
# -----------------------------

class SoftmaxRegression:
    def __init__(self, in_dim: int, num_classes: int):
        self.W = np.zeros((in_dim, num_classes), dtype=np.float64)
        self.b = np.zeros((num_classes,), dtype=np.float64)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_parameters(self, params: Dict[str, np.ndarray]):
        self.W = params["W"].copy()
        self.b = params["b"].copy()

    def forward(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.W + self.b
        z -= z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        probs = exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)
        return probs

    def loss_and_gradients(self, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> Tuple[float, Dict[str, np.ndarray]]:
        n = X.shape[0]
        probs = self.forward(X)
        y_onehot = one_hot(y, probs.shape[1])
        # Cross-entropy
        eps = 1e-12
        ce = -np.sum(y_onehot * np.log(probs + eps)) / n
        # L2 regularization
        reg = 0.5 * l2 * np.sum(self.W * self.W)
        loss = ce + reg
        # Gradients
        dZ = (probs - y_onehot) / n
        dW = X.T @ dZ + l2 * self.W
        db = np.sum(dZ, axis=0)
        return loss, {"W": dW, "b": db}

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# -----------------------------
# Client logic: local SGD epochs
# -----------------------------

class Client:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray):
        self.cid = cid
        self.X = X
        self.y = y

    def num_samples(self) -> int:
        return self.X.shape[0]

    def local_train(self,
                    global_params: Dict[str, np.ndarray],
                    in_dim: int,
                    num_classes: int,
                    lr: float,
                    batch_size: int,
                    local_epochs: int,
                    l2: float,
                    seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        model = SoftmaxRegression(in_dim, num_classes)
        model.set_parameters(global_params)

        X, y = shuffle_data(self.X, self.y, seed)
        n = X.shape[0]
        steps_per_epoch = math.ceil(n / batch_size)

        for _ in range(local_epochs):
            X, y = shuffle_data(X, y, seed)
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = min(n, start + batch_size)
                xb = X[start:end]
                yb = y[start:end]
                _, grads = model.loss_and_gradients(xb, yb, l2=l2)
                model.W -= lr * grads["W"]
                model.b -= lr * grads["b"]
        return model.parameters()


# -----------------------------
# Aggregation: FedAvg (weighted by samples)
# -----------------------------

def fedavg( client_params: List[Dict[str, np.ndarray]], client_sizes: List[int]) -> Dict[str, np.ndarray]:
    total = float(sum(client_sizes))
    assert total > 0
    keys = client_params[0].keys()
    agg = {k: np.zeros_like(client_params[0][k]) for k in keys}
    for params, n in zip(client_params, client_sizes):
        w = n / total
        for k in keys:
            agg[k] += w * params[k]
    return agg


# -----------------------------
# Data loading
# -----------------------------

def load_clients_from_npz(root: str) -> Tuple[List[Client], int, int]:
    paths = sorted(glob.glob(os.path.join(root, "client_*.npz")))
    if not paths:
        raise FileNotFoundError("No client_*.npz files found")

    X0 = None
    num_classes = None
    clients: List[Client] = []

    for i, p in enumerate(paths):
        data = np.load(p)
        X = data["X"].astype(np.float64)
        y = data["y"].astype(int)
        if X0 is None:
            X0 = X
            in_dim = X.shape[1]
        else:
            in_dim = X0.shape[1]
            if X.shape[1] != in_dim:
                raise ValueError(f"Inconsistent feature dims: {p}")
        if num_classes is None:
            num_classes = int(np.max(y)) + 1
        else:
            num_classes = max(num_classes, int(np.max(y)) + 1)
        clients.append(Client(i, X, y))

    assert X0 is not None and num_classes is not None
    return clients, in_dim, num_classes


def synthesize_clients(num_clients: int = 5,
                       samples_per_client: int = 1000,
                       in_dim: int = 20,
                       num_classes: int = 3,
                       heterogeneity: float = 2.0,
                       seed: int = 42) -> Tuple[List[Client], int, int]:
    set_seed(seed)
    clients: List[Client] = []
    # Non-IID via client-specific class priors and feature shifts
    class_means = np.random.randn(num_classes, in_dim) * 2.0
    for cid in range(num_clients):
        priors = np.random.dirichlet(alpha=np.ones(num_classes) * heterogeneity)
        X_list = []
        y_list = []
        for _ in range(samples_per_client):
            c = np.random.choice(num_classes, p=priors)
            x = class_means[c] + np.random.randn(in_dim) * 1.0 + np.random.randn(in_dim) * (0.2 * cid)
            X_list.append(x)
            y_list.append(c)
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=int)
        clients.append(Client(cid, X, y))
    return clients, in_dim, num_classes


def load_fedkits_or_synthetic(root: Optional[str],
                              fallback_clients: int = 5,
                              fallback_samples: int = 800,
                              seed: int = 42) -> Tuple[List[Client], int, int]:
    if root and os.path.isdir(root):
        try:
            clients, in_dim, num_classes = load_clients_from_npz(root)
            return clients, in_dim, num_classes
        except Exception as e:
            print(f"[Data] Could not load NPZ clients from {root}: {e}. Falling back to synthetic.")
    return synthesize_clients(num_clients=fallback_clients,
                              samples_per_client=fallback_samples,
                              in_dim=20,
                              num_classes=3,
                              heterogeneity=2.0,
                              seed=seed)


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_global(model_params: Dict[str, np.ndarray],
                    clients: List[Client],
                    in_dim: int,
                    num_classes: int,
                    sample_cap_per_client: Optional[int] = 200) -> Dict[str, float]:
    model = SoftmaxRegression(in_dim, num_classes)
    model.set_parameters(model_params)

    Xs = []
    ys = []
    for c in clients:
        n = c.num_samples()
        take = min(n, sample_cap_per_client) if sample_cap_per_client else n
        idx = np.random.permutation(n)[:take]
        Xs.append(c.X[idx])
        ys.append(c.y[idx])
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    pred = model.predict(X)
    acc = float(np.mean(pred == y))
    return {"accuracy": acc}


# -----------------------------
# Federated training loop
# -----------------------------

def run_fedavg(
    root: Optional[str] = None,
    num_rounds: int = 20,
    fraction_clients: float = 0.6,
    lr: float = 0.5,
    batch_size: int = 64,
    local_epochs: int = 2,
    l2: float = 1e-4,
    seed: int = 42,
    eval_every: int = 1,
) -> Dict[str, float]:
    set_seed(seed)
    clients, in_dim, num_classes = load_fedkits_or_synthetic(root, seed=seed)

    # Initialize global params
    model = SoftmaxRegression(in_dim, num_classes)
    global_params = model.parameters()

    history = []

    for rnd in range(1, num_rounds + 1):
        start = time.time()
        m = max(1, int(math.ceil(fraction_clients * len(clients))))
        selected = np.random.choice(len(clients), size=m, replace=False)

        client_params = []
        client_sizes = []
        for idx in selected:
            c = clients[idx]
            params = c.local_train(
                global_params=global_params,
                in_dim=in_dim,
                num_classes=num_classes,
                lr=lr,
                batch_size=batch_size,
                local_epochs=local_epochs,
                l2=l2,
                seed=seed + rnd + c.cid,
            )
            client_params.append(params)
            client_sizes.append(c.num_samples())

        global_params = fedavg(client_params, client_sizes)
        dur = time.time() - start

        if (rnd % eval_every) == 0:
            metrics = evaluate_global(global_params, clients, in_dim, num_classes)
            metrics["round"] = rnd
            metrics["duration_sec"] = dur
            history.append(metrics)
            print(f"[Round {rnd:03d}] acc={metrics['accuracy']:.4f} time={dur:.2f}s")

    if history:
        print("Final:", history[-1])
        return history[-1]
    return {"accuracy": 0.0}


# -----------------------------
# CLI / Main
# -----------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pure-NumPy FedAvg (Softmax Regression)")
    parser.add_argument("--root", type=str, default=None,
                        help="Fed-KITS root with client_*.npz (X,y). If absent, uses synthetic data.")
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--fraction_clients", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_metrics", type=str, default=None, help="Path to save final metrics JSON")

    args = parser.parse_args()

    final = run_fedavg(
        root=args.root,
        num_rounds=args.num_rounds,
        fraction_clients=args.fraction_clients,
        lr=args.lr,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        l2=args.l2,
        seed=args.seed,
        eval_every=args.eval_every,
    )

    if args.save_metrics:
        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        print(f"Saved metrics to {args.save_metrics}")


if __name__ == "__main__":
    main()
