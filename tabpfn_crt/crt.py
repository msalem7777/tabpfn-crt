import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor, TabPFNClassifier
from tabpfn.constants import ModelVersion

from .utils import is_categorical, logp_from_full_output


def tabpfn_crt(
    X,
    y,
    j,
    *,
    B=200,
    alpha=0.05,
    test_size=0.2,
    seed=0,
    device=None,
    K=100,
    max_unique_cat=10,
):
    """
    Conditional Randomization Test (CRT) using TabPFN as a fixed predictive model.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Feature matrix.
    y : array-like, shape (n,)
        Target variable.
    j : int
        Index of feature to test.
    B : int
        Number of CRT resamples.
    alpha : float
        Significance level.
    K : int
        Number of quantiles for continuous conditional sampling.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.RandomState(seed)

    # ---------------------------
    # Train / evaluation split
    # ---------------------------
    X_tr, X_ev, y_tr, y_ev = train_test_split(
        np.asarray(X),
        np.asarray(y),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    # ---------------------------
    # Fit predictive model for y | X
    # ---------------------------
    y_is_cat = is_categorical(y_tr, max_unique_cat)
    ModelY = TabPFNClassifier if y_is_cat else TabPFNRegressor

    model_y = ModelY.create_default_for_version(
        ModelVersion.V2,
        device=device,
    )
    model_y.fit(X_tr, y_tr)

    full_plus = model_y.predict(X_ev, output_type="full")
    logp_plus = logp_from_full_output(full_plus, y_ev)
    T_obs = np.mean(logp_plus)

    # ---------------------------
    # Fit conditional model for X_j | X_-j
    # ---------------------------
    Xm_tr = np.delete(X_tr, j, axis=1)
    Xm_ev = np.delete(X_ev, j, axis=1)
    xj_tr = X_tr[:, j]

    xj_is_cat = is_categorical(xj_tr, max_unique_cat)
    ModelXJ = TabPFNClassifier if xj_is_cat else TabPFNRegressor

    model_xj = ModelXJ.create_default_for_version(
        ModelVersion.V2,
        device=device,
    )
    model_xj.fit(Xm_tr, xj_tr)

    # ---------------------------
    # Precompute conditional sampler
    # ---------------------------
    if not xj_is_cat:
        quantiles = np.linspace(0, 1, K)
        Q = model_xj.predict(
            Xm_ev,
            output_type="quantiles",
            quantiles=quantiles,
        )
        if Q.shape[0] != K:
            Q = Q.T  # ensure (K, n_eval)

    # ---------------------------
    # CRT null distribution
    # ---------------------------
    n_ev = X_ev.shape[0]
    T_null = np.zeros(B)

    for b in range(B):
        if xj_is_cat:
            probs = model_xj.predict_proba(Xm_ev)
            xj_null = np.array([
                rng.choice(model_xj.classes_, p=probs[i])
                for i in range(n_ev)
            ])
        else:
            idx = rng.randint(0, K, size=n_ev)
            xj_null = Q[idx, np.arange(n_ev)]

            bad = ~np.isfinite(xj_null)
            tries = 0
            while bad.any():
                if tries >= 10:
                    raise RuntimeError("Non-finite CRT samples after retries")
                idx_bad = rng.randint(0, K, size=bad.sum())
                xj_null[bad] = Q[idx_bad, np.where(bad)[0]]
                bad = ~np.isfinite(xj_null)
                tries += 1

        X_ev_null = X_ev.copy()
        X_ev_null[:, j] = xj_null

        full_null = model_y.predict(X_ev_null, output_type="full")
        logp_null = logp_from_full_output(full_null, y_ev)
        T_null[b] = np.mean(logp_null)

    # ---------------------------
    # p-value
    # ---------------------------
    p_value = (1 + np.sum(T_null >= T_obs)) / (B + 1)

    return {
        "p_value": float(p_value),
        "reject_null": bool(p_value <= alpha),
        "alpha": alpha,
        "T_obs": float(T_obs),
        "T_null": T_null,
        "y_is_categorical": y_is_cat,
        "xj_is_categorical": xj_is_cat,
        "B": B,
        "K": None if xj_is_cat else K,
        "j": int(j),
    }
