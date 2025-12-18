import numpy as np
import torch


def is_categorical(arr, max_unique=10):
    """
    Heuristic to determine whether a variable should be treated as categorical.

    Parameters
    ----------
    arr : array-like
        Input array.
    max_unique : int
        Maximum number of unique values for categorical treatment.

    Returns
    -------
    bool
        True if categorical, False otherwise.
    """
    arr = np.asarray(arr)
    uniq = np.unique(arr[~np.isnan(arr)])
    return len(uniq) < max_unique


def logp_from_full_output(full_out, y_np):
    """
    Extract log predictive density from TabPFN 'full' prediction output.

    Parameters
    ----------
    full_out : dict
        Output from TabPFN predict(..., output_type="full").
    y_np : array-like
        Ground-truth targets.

    Returns
    -------
    np.ndarray
        Log predictive density for each observation.
    """
    criterion = full_out["criterion"]
    logits = full_out["logits"]

    y_torch = torch.as_tensor(
        y_np,
        device=logits.device,
        dtype=logits.dtype,
    ).view(*logits.shape[:-1])

    nll = criterion(logits, y_torch)
    return (-nll).detach().cpu().numpy().reshape(-1)
