import os
import numpy as np


def softmax(X, theta=1.0, axis=None) -> np.ndarray:
    """
    Compute the softmax of each element along an axis of X.

    Parameters:
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns:
    --------
    p: np.ndarray
        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def directory_chunker(d: str, n: int):
    """
    Chunk a directory of files into lists
    of length n.

    Parameters:
    -----------
    d: str
    n: int
    """
    files = os.listdir(d)
    n_files = len(files)
    counter = 0
    done = False
    while not done:
        si, ei = counter * n, (counter + 1) * n
        ei = min(ei, n_files)
        if ei == n_files:
            done = True
        chunk = files[si:ei]
        counter += 1
        yield chunk
