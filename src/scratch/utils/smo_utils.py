import numpy as np


def compute_error(model, X, y, i):
    """
    Compute the error for the i-th training example.

    Parameters
    ----------
    model : SVM
        The SVM model instance.
    X : ndarray, shape (n_samples, n_features)
        Training feature data.
    y : ndarray, shape (n_samples,)
        Training labels.
    i : int
        Index of the example.

    Returns
    -------
    float
        Error E_i = f(x_i) - y_i, where f(x_i) is the decision function.
    """
    f_i = (
        sum(model.alphas[j] * y[j] * model._kernel(X[j], X[i]) for j in range(len(y)))
        + model.b
    )
    return f_i - y[i]


def select_second_alpha(model, i, E_i):
    """
    Select the second alpha index j to maximize the step size |E_i - E_j|.

    Parameters
    ----------
    model : SVM
        The SVM model instance.
    i : int
        Index of the first alpha.
    E_i : float
        Error for the first example.

    Returns
    -------
    int
        Index j of the second alpha, or -1 if no suitable j is found.
    """
    max_delta = 0
    j = -1
    for k in range(len(model.alphas)):
        if k == i:
            continue
        E_k = model.error_cache[k]
        delta = abs(E_i - E_k)
        if delta > max_delta:
            max_delta = delta
            j = k
    return j


def compute_L_H(alpha_i, alpha_j, y_i, y_j, C):
    """
    Compute the lower (L) and upper (H) bounds for alpha_j.

    Parameters
    ----------
    alpha_i : float
        Current value of the first alpha.
    alpha_j : float
        Current value of the second alpha.
    y_i : int
        Label of the first example (-1 or 1).
    y_j : int
        Label of the second example (-1 or 1).
    C : float
        Regularization parameter.

    Returns
    -------
    tuple
        (L, H) bounds for alpha_j.
    """
    if y_i != y_j:
        L = max(0, alpha_j - alpha_i)
        H = min(C, C + alpha_j - alpha_i)
    else:
        L = max(0, alpha_i + alpha_j - C)
        H = min(C, alpha_i + alpha_j)
    return L, H


def clip_alpha_j(alpha_j_new, L, H):
    """
    Clip the new alpha_j value to the interval [L, H].

    Parameters
    ----------
    alpha_j_new : float
        Unclipped new value for alpha_j.
    L : float
        Lower bound.
    H : float
        Upper bound.

    Returns
    -------
    float
        Clipped value of alpha_j.
    """
    if alpha_j_new > H:
        return H
    elif alpha_j_new < L:
        return L
    return alpha_j_new
