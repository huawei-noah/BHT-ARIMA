import warnings
import scipy
import numpy as np


def svd_fun(matrix, n_eigenvecs=None):
    """Computes a fast partial SVD on `matrix`
    If `n_eigenvecs` is specified, sparse eigendecomposition is used on
    either matrix.dot(matrix.T) or matrix.T.dot(matrix).
    Parameters
    ----------
    matrix : tensor
        A 2D tensor.
    n_eigenvecs : int, optional, default is None
        If specified, number of eigen[vectors-values] to return.
    Returns
    -------
    U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
        Contains the right singular vectors
    S : 1-D tensor, shape (n_eigenvecs, )
        Contains the singular values of `matrix`
    V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
        Contains the left singular vectors
    """

    # Choose what to do depending on the params
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
        max_dim = dim_2
    else:
        min_dim = dim_2
        max_dim = dim_1

    if n_eigenvecs >= min_dim:
        if n_eigenvecs > max_dim:
            warnings.warn(('Trying to compute SVD with n_eigenvecs={0}, which '
                           'is larger than max(matrix.shape)={1}. Setting '
                           'n_eigenvecs to {1}').format(n_eigenvecs, max_dim))
            n_eigenvecs = max_dim

        if n_eigenvecs is None or n_eigenvecs > min_dim:
            full_matrices = True
        else:
            full_matrices = False

        # Default on standard SVD
        U, S, V = scipy.linalg.svd(matrix, full_matrices=full_matrices)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
    else:
        # We can perform a partial SVD
        # First choose whether to use X * X.T or X.T *X
        if dim_1 < dim_2:
            S, U = scipy.sparse.linalg.eigsh(
                np.dot(matrix, matrix.T.conj()), k=n_eigenvecs, which='LM'
            )
            S = np.sqrt(S)
            V = np.dot(matrix.T.conj(), U * 1 / S[None, :])
        else:
            S, V = scipy.sparse.linalg.eigsh(
                np.dot(matrix.T.conj(), matrix), k=n_eigenvecs, which='LM'
            )
            S = np.sqrt(S)
            U = np.dot(matrix, V) * 1 / S[None, :]

        # WARNING: here, V is still the transpose of what it should be
        U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        V = V.T.conj()
    return U, S, V