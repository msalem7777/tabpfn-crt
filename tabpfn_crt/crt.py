import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor, TabPFNClassifier
from tabpfn.constants import ModelVersion

from .utils import is_categorical, logp_from_full_output, logp_from_proba

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
    # Split
    # ---------------------------
    X_tr, X_ev, y_tr, y_ev = train_test_split(
        np.asarray(X),
        np.asarray(y),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    # ---------------------------
    # Model for y | X
    # ---------------------------
    y_is_cat = is_categorical(y_tr, max_unique_cat)
    ModelY = TabPFNClassifier if y_is_cat else TabPFNRegressor

    model_y = ModelY.create_default_for_version(
        ModelVersion.V2,
        device=device,
        fit_mode="fit_with_cache",
    )
    model_y.fit(X_tr, y_tr)

    if y_is_cat:
        probs_plus = model_y.predict_proba(X_ev)
        logp_plus = logp_from_proba(probs_plus, y_ev, model_y.classes_)
    else:
        full_plus = model_y.predict(X_ev, output_type="full")
        logp_plus = logp_from_full_output(full_plus, y_ev)

    # ---------------------------
    # Observed T_obs
    # ---------------------------
    T_obs = np.mean(logp_plus)

    # ---------------------------
    # Model for Xj | X_-j
    # ---------------------------
    Xm_tr = np.delete(X_tr, j, axis=1)
    Xm_ev = np.delete(X_ev, j, axis=1)
    xj_tr = X_tr[:, j]

    xj_is_cat = is_categorical(xj_tr, max_unique_cat)
    ModelXJ = TabPFNClassifier if xj_is_cat else TabPFNRegressor

    model_xj = ModelXJ.create_default_for_version(
        ModelVersion.V2,
        device=device,
        fit_mode="fit_with_cache",
    )
    model_xj.fit(Xm_tr, xj_tr)

    # ---------------------------
    # Precompute conditional sampler
    # ---------------------------
    if not xj_is_cat:
        q_grid = np.linspace(0, 1, K)
        Q = np.asarray(
            model_xj.predict(
                Xm_ev,
                output_type="quantiles",
                quantiles=q_grid,
            )
        )
        if Q.shape[0] != K:
            Q = Q.T  # ensure (K, n_ev)
    elif xj_is_cat:
        probs = model_xj.predict_proba(Xm_ev)
        cdf = np.cumsum(probs, axis=1)

        if not np.all(np.isfinite(probs)):
            i = np.argwhere(~np.isfinite(probs))[0][0]
            print("Non-finite probs at row", i, probs[i])
            raise ValueError("bad probs")
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            i = np.argmax(np.abs(row_sums - 1.0))
            print("Bad prob sum at row", i, row_sums[i], probs[i])
            raise ValueError("probabilities not normalized")

    # ---------------------------
    # Null distribution
    # ---------------------------
    T_null = np.zeros(B)
    n_ev = X_ev.shape[0]
    X_ev_null = X_ev.copy()

    for b in range(B):
        if xj_is_cat:
            u = rng.rand(n_ev, 1)
            idx = (u <= cdf).argmax(axis=1)
            xj_null = model_xj.classes_[idx]
        else:
            idx = rng.randint(0, K, size=n_ev)
            xj_null = Q[idx, np.arange(n_ev)]

            bad = ~np.isfinite(xj_null)
            max_resample = 10
            n_try = 0

            while bad.any():
                if n_try >= max_resample:
                    raise RuntimeError(
                        f"CRT quantile sampling produced non-finite values after {max_resample} retries"
                    )

                # resample ONLY the bad positions
                idx_bad = rng.randint(0, K, size=bad.sum())
                xj_null[bad] = Q[idx_bad, np.where(bad)[0]]

                bad = ~np.isfinite(xj_null)
                n_try += 1

        X_ev_null[:, j] = np.asarray(xj_null)
        
        if y_is_cat:
            probs_null = model_y.predict_proba(X_ev_null)
            logp_null = logp_from_proba(probs_null, y_ev, model_y.classes_)
        else:
            full_null = model_y.predict(X_ev_null, output_type="full")
            logp_null = logp_from_full_output(full_null, y_ev)

        T_null[b] = np.mean(logp_null)
    # ---------------------------
    # p-value (right-tailed)
    # ---------------------------
    p_value = float((1 + np.sum(T_null >= T_obs)) / (B + 1))

    # ---------------------------
    # Human-readable interpretation
    # ---------------------------
    reject = p_value <= alpha

    if reject:
        relevance_stmt = (
            f"Result: REJECT H0 at α = {alpha:.2f}.\n"
            f"Interpretation: The variable X[{j}] provides information about the "
            f"target Y that is not explained by the remaining covariates."
        )
    else:
        relevance_stmt = (
            f"Result: FAIL TO REJECT H0 at α = {alpha:.2f}.\n"
            f"Interpretation: There is no evidence that X[{j}] provides additional "
            f"information about the target Y beyond the remaining covariates."
        )

    summary_stmt = (
        "\n=== Conditional Randomization Test (TabPFN) ===\n"
        f"p-value: {p_value:.4g}\n"
        f"{relevance_stmt}"
    )

    print(summary_stmt)

    return {
        "p_value": float(p_value),
        "reject_null": bool(reject),
        "alpha": alpha,
        "T_obs": T_obs,
        "T_null": T_null,
        "interpretation": relevance_stmt,
        "y_is_categorical": y_is_cat,
        "xj_is_categorical": xj_is_cat,
        "B": B,
        "K": K if not xj_is_cat else None,
        "j": int(j),
    }