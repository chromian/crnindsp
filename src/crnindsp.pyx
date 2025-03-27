# cython: language_level = 3
# cython: language = c++

# =========================================================================================================== #
import cython
cimport numpy as cnp
import numpy as np
from libc.math cimport fabs
from scipy.linalg.cython_lapack cimport dgetrf
from cython.parallel  import prange
from scipy.optimize import minimize
# =========================================================================================================== #

cdef double InF = <double>(float('inf'))
cdef double NaN = <double>(float('nan'))
cdef double tol = 1e-9

cdef inline int iszero(double x) nogil:
    return (fabs(x) < tol)

cdef void swap_rows(cnp.ndarray[double, ndim=2] mat, int i, int j):
    """ Swap rows i and j in matrix mat. """
    cdef:
        int cols = mat.shape[1]
        double[:] temp = np.copy(mat[i])
    mat[i] = mat[j]
    mat[j] = temp

cdef cnp.ndarray[double, ndim=2] cy_rref(cnp.ndarray[double, ndim=2] mat):
    """ Compute the reduced row echelon form of a given matrix. """
    cdef:
        int rows = mat.shape[0]
        int cols = mat.shape[1]
        int col = 0, row = 0, i
        double pivot
    # Work on a copy to avoid modifying the original matrix
    mat = mat.copy()
    while (col < cols) and (row < rows):
        pivot_row = row
        for i in range(row + 1, rows):
            if fabs(mat[i, col]) > fabs(mat[pivot_row, col]):
                pivot_row = i
        if mat[pivot_row, col] == 0:
            col += 1
            continue
        # Swap rows to move pivot row to the current row
        if pivot_row != row:
            swap_rows(mat, row, pivot_row)
        # Normalize the pivot row
        pivot = mat[row, col]
        mat[row] /= pivot
        # Eliminate entries above and below the pivot (reduction)
        for i in range(rows):
            if i != row:
                mat[i] -= mat[row] * mat[i, col]
        row += 1
        col += 1
    return mat

cdef tuple cy_kernel_image(cnp.ndarray[double, ndim=2] mat):
    """ Compute the null space and the image space of a given matrix using Gaussian elimination. """
    cdef:
        int rows = mat.T.shape[0]
        int cols = mat.T.shape[1]
        cnp.ndarray[double, ndim=2] gaussian = np.zeros(shape = (rows, cols + rows))
        tuple res
    gaussian[:,:cols] = mat.T
    gaussian[:,cols:] = np.eye(rows)
    gaussian = cy_rref(gaussian)
    KIindex = np.all(np.abs(gaussian[:,:cols]) < tol, axis = 1)
    res = (gaussian[ KIindex,cols:].T, gaussian[~KIindex,:cols].T)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cydet(double[:,:] MAT, int n, int[:] IPIV, int INFO) nogil:
    """ Compute the determinant of a square matrix MAT using  DGETRF from the LAPACK library. """
    cdef int j
    cdef double detval = 1.0
    dgetrf(&n, &n, &MAT[0,0], &n, &IPIV[0], &INFO)
    for j in range(n):
        if j != (IPIV[j] - 1):
            detval *= -(MAT[j, j])
        else:
            detval *= MAT[j, j]
    return detval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] cyadjugate(double[:,:] mat, int n, double[:,:,:] WORK, int[:,:] IPIV) nogil:
    """ Compute the adjugate matrix of a square matrix mat. """
    cdef int i, j, row, col, nm1 = n-1
    for row in prange(n):
        for col in prange(n):
            for i in prange(nm1):
                for j in prange(nm1):
                    if (i>=row) and (j>=col):
                        WORK[n*row+col,i,j] = mat[i+1,j+1]
                    elif j >= col:
                        WORK[n*row+col,i,j] = mat[i,j+1]
                    elif i >= row:
                        WORK[n*row+col,i,j] = mat[i+1,j]
                    else:
                        WORK[n*row+col,i,j] = mat[i,j]
    for row in prange(n):
        for col in prange(n):
            mat[col, row] = (-1)**(row+col) * cydet(WORK[n*row+col], n-1, IPIV[n*row+col], 0)
    return mat

def adj(A):
    """ Compute the adjugate matrix of a square matrix. """
    n = A.shape[0]
    WORK = np.empty(shape = (n*n, n-1, n-1), dtype = float)
    IPIV = np.zeros(shape = (n*n, n-1), dtype = np.int32)
    return cyadjugate(A.copy(), n, WORK, IPIV)

def det(A):
    """ Compute the determinant of a square matrix. """
    return cydet(A.copy(), A.shape[0], np.zeros(A.shape[0], dtype=np.int32), 1)

def det_for_symM(A, N=10, nlogtol=9):
    """
    Determines the sign of the determinant of a matrix A with placeholders (inf or NaN),
    using numerical optimization.

    Parameters:
        A (numpy.ndarray): Input matrix with placeholders.
        N (int): Number of optimization attempts (default: 10).
        nlogtol (int): Logarithmic tolerance for rounding.

    Returns:
        float: -1.0 if determinant is consistently negative,
                1.0 if determinant is consistently positive,
                0.0 if determinant is always zero,
                NaN if determinant changes sign.
    """
    # Identify placeholder positions and their signs
    placeholder_mask = np.isinf(A) | np.isnan(A)
    placeholder_signs = np.sign(A[placeholder_mask])
    num_vars = len(placeholder_signs)
    if num_vars == 0:
        return np.sign(np.linalg.det(A))
    def compute_det(x):
        # evaluate determinant by replacing placeholders with values in x.
        x = np.array(x)
        tol = 10 ** (-nlogtol)
        # assign values based on placeholder signs
        x[placeholder_signs == 1]  = x[placeholder_signs == 1] ** 2
        x[placeholder_signs == -1] = -x[placeholder_signs == -1] ** 2
        x[np.abs(placeholder_signs) < tol] = x[np.abs(placeholder_signs) < tol] ** 3
        # replace placeholders and compute determinant
        varm = A.copy()
        varm[placeholder_mask] = x
        return np.arctan(det(varm))
    # Perform N optimization attempts
    opt_neg = [minimize(lambda x: compute_det(x),
                        np.random.normal(size=num_vars, scale = 20)).fun for _ in range(N)]
    opt_pos = [-minimize(lambda x: -compute_det(x),
                         np.random.normal(size=num_vars, scale = 20)).fun for _ in range(N)]
    has_negative = np.min(np.sign(opt_neg)) < 0
    has_positive = np.max(np.sign(opt_pos)) > 0
    if has_negative and has_positive:
        return np.nan
    elif has_positive:
        return  1.0
    elif has_negative:
        return -1.0
    else:
        return 0

cdef cnp.ndarray[double, ndim=2] construct_A_from(cnp.ndarray[double, ndim=2] stoi,
                                                  cnp.ndarray[double, ndim=2] reginfo):
    """ Construct the A matrix from stoichiometric and regulation information. """
    C = cy_kernel_image(stoi)[0]
    DT = cy_kernel_image(stoi.T)[0].T
    M = stoi.shape[0]
    N = stoi.shape[1]
    A = np.zeros((M + C.shape[1], M + C.shape[1]))
    A[:N, :M] = reginfo.T
    A[:N, M:] = C
    A[N:, :M] = DT
    return A

cdef int _check_OC(double[:,:] reg_info, list X, list R):
    """ Verify output-completeness of a subnetwork. """
    cdef int m, n
    for m in X:
        if any(not iszero(reg_info[m, n]) for n in set(range(reg_info.shape[1])) - set(R)):
            return False
    return True

cdef int iszero1d(x):
    """ Check if all elements in an array are within tolerance of zero. """
    return np.all(np.abs(x) < tol)

cdef _check_no_eCQs(cnp.ndarray[double, ndim=2] stoi, cnp.ndarray[double, ndim=2] DT, list X, list R):
    """ Verify that the subnetwork has no emergent conserved quantities. """
    X_mask, R_mask = np.isin(range(stoi.shape[0]), X), np.isin(range(stoi.shape[1]), R)
    C11 = cy_kernel_image(stoi[X_mask, :][:, R_mask])[0]
    NU01C11 = np.dot(stoi[~X_mask, :][:, R_mask], C11)
    DT_proj = np.dot(DT[:, X_mask], stoi[X_mask, :][:, R_mask])
    return (iszero1d(NU01C11) and iszero1d(DT_proj))

cdef tuple _compute_index_and_det(cnp.ndarray[double, ndim=2] stoi,
                                  cnp.ndarray[double, ndim=2] reg_info,
                                  list X, list R):
    """ Compute index and determinant sign for a subnetwork. """
    cdef cnp.ndarray[double, ndim=2] subA
    X_mask, R_mask = np.isin(range(stoi.shape[0]), X), np.isin(range(stoi.shape[1]), R)
    C11   = cy_kernel_image(stoi[X_mask, :][:, R_mask])[0]
    D     = cy_kernel_image(stoi.T)[0]
    Dproj = cy_kernel_image(D[X_mask, :])[1]
    index = - len(X) + len(R) - C11.shape[1] + Dproj.shape[1]
    if index == 0:
        subA = construct_A_from(stoi[X_mask, :][:, R_mask], reg_info[X_mask, :][:, R_mask])
        return (index, det_for_symM(subA))
    return (index, None)


class CRN:
    """ Class representing a Chemical Reaction Network (CRN). """

    def __init__(self, stoi, reg_info = None, X_names = None, R_names = None, name = None):
        self.name = name
        self._stoi = np.array(stoi, dtype=float)
        self._X_names = X_names or [f"X{m}" for m in range(stoi.shape[0])]
        self._R_names = R_names or [f"R{n}" for n in range(stoi.shape[1])]

        if not (reg_info is None):
            self._reg_info = np.array(reg_info, dtype=float)
        else:
            self._reg_info = np.where(stoi < tol, -np.inf, 0.0)

        self._saved_S = None

    def A_mat(self):
        return construct_A_from(self._stoi, self._reg_info)

    def check_oc(self, X, R):
        return bool(_check_OC(self._reg_info, X, R))

    def check_no_ecqs(self, X, R):
        return bool(_check_no_eCQs(self._stoi, cy_kernel_image(self._stoi.T)[0].T, X, R))

    def compute_idx_and_det(self, X, R):
        return _compute_index_and_det(self._stoi, self._reg_info, X, R)

    def _smat(self, N = 10):
        """sensitivity table dx/dr"""
        A = (self.A_mat()).copy()
        entryholder = ~(np.abs(A) < np.inf)
        S = (self._stoi * 0) > 1
        for n in range(N):
            A[entryholder] = np.random.normal(size  = (np.sum(entryholder),),
                                              scale = 10)
            _S = np.abs(adj(A)[:self._stoi.shape[0], :self._stoi.shape[1]]) > tol
            S  = S + _S
        return S

    def _compute_BS(self, startfrom = 'R', N = 10):
        BSs = []
        DT_bool  = (np.abs(cy_kernel_image(self._stoi.T)[0].T) > tol)
        self._saved_S = self._smat(N = N)
        if startfrom == 'R':
            for n in range(self._stoi.shape[1]):
                Xids, Rids = [], [n]
                while True:
                    X0s = [int(i) for i in np.arange(self._saved_S.shape[0])[np.any(self._saved_S[:, Rids], axis = 1)]]
                    Xs = [int(i) for i in np.arange(self._saved_S.shape[0])[np.any(DT_bool[np.any(DT_bool[:, X0s], axis = 1), :], axis = 0)]]
                    for x in X0s:
                        Xs.append(int(x))
                    for xids in (set(Xs) - set(Xids)):
                        Xids.append(xids)
                    Xids = list(set(Xids))
                    if self.check_oc(Xids, Rids):
                        BSs.append(([int(m) for m in np.sort(Xids)], [int(n) for n in np.sort(Rids)],))
                        break
                    else:
                        Rs = np.any(~(np.abs(self._reg_info[Xids, :]) < tol), axis = 0)
                        Rs = np.arange(len(Rs))[Rs]
                        for rids in (set(Rs) - set(Rids)):
                            Rids.append(int(rids))
        elif startfrom == 'X':
            for m in range(self._stoi.shape[0]):
                Xids, Rids = [m], []
                for _m in np.arange(self._saved_S.shape[0])[np.any(DT_bool[DT_bool[:, m], :], axis = 0)]:
                    Xids.append(int(_m))
                Xids = list(set(Xids))
                while True:
                    Rs = np.any(~(np.abs(self._reg_info[Xids, :]) < tol), axis = 0)
                    Rs = np.arange(len(Rs))[Rs]
                    for rids in (set(Rs) - set(Rids)):
                        Rids.append(int(rids))
                    X0s = [int(i) for i in np.arange(self._saved_S.shape[0])[np.any(self._saved_S[:, Rids], axis = 1)]]
                    Xs = [int(i) for i in np.arange(self._saved_S.shape[0])[np.any(DT_bool[np.any(DT_bool[:, X0s], axis = 1), :], axis = 0)]]
                    for x in X0s:
                        Xs.append(int(x))
                    for xids in (set(Xs) - set(Xids)):
                        Xids.append(xids)
                    if self.check_oc(Xids, Rids):
                        BSs.append(([int(m) for m in np.sort(Xids)], [int(n) for n in np.sort(Rids)],))
                        break
                    else:
                        pass
        elif startfrom == 'XR':
            for n in range(self._stoi.shape[1]):
                for m in range(self._stoi.shape[0]):
                    Xids, Rids = [m], [n]
                    for _m in np.arange(self._saved_S.shape[0])[np.any(DT_bool[DT_bool[:, m], :], axis = 0)]:
                        Xids.append(int(_m))
                    Xids = list(set(Xids))
                    while True:
                        X0s = [int(i) for i in np.arange(self._saved_S.shape[0])[np.any(self._saved_S[:, Rids], axis = 1)]]
                        Xs = [int(i) for i in np.arange(self._saved_S.shape[0])[np.any(DT_bool[np.any(DT_bool[:, X0s], axis = 1), :], axis = 0)]]
                        for x in X0s:
                            Xs.append(int(x))
                        for xids in (set(Xs) - set(Xids)):
                            Xids.append(xids)
                        if self.check_oc(Xids, Rids):
                            BSs.append(([int(m) for m in np.sort(Xids)], [int(n) for n in np.sort(Rids)],))
                            break
                        else:
                            Rs = np.any(~(np.abs(self._reg_info[Xids, :]) < tol), axis = 0)
                            Rs = np.arange(len(Rs))[Rs]
                            for rids in (set(Rs) - set(Rids)):
                                Rids.append(int(rids))
        order = np.argsort([len(bs[0]) for bs in BSs])
        BSs   = [BSs[order[id]] for id in range(len(order))]
        uBSs  = []
        # to remove duplicates
        for g in BSs:
            flag_unique = True
            for g_ in uBSs:
                if g == g_:
                    flag_unique = False
                    break
            if flag_unique:
                uBSs.append(g)
        return uBSs

    def structural_reduction(self, X, R):
        X_mask = np.isin(range(self._stoi.shape[0]), X)
        R_mask = np.isin(range(self._stoi.shape[1]), R)
        NU11 = self._stoi[ X_mask, :][ :, R_mask]
        NU10 = self._stoi[ X_mask, :][ :,~R_mask]
        NU01 = self._stoi[~X_mask, :][ :, R_mask]
        NU00 = self._stoi[~X_mask, :][ :,~R_mask]
        NU_rd = NU00 - np.dot(np.dot(NU01, np.linalg.pinv(NU11)), NU10)
        # start updating
        self._stoi = NU_rd
        self._reg_info = self._reg_info[~X_mask, :][ :,~R_mask]
        self._X_names  = [self._X_names[m] for m in range(len(self._X_names)) if (not X_mask[m])]
        self._R_names  = [self._R_names[n] for n in range(len(self._R_names)) if (not R_mask[n])]
        return None

    def proper_mBS(self, modes = ['R', 'X', 'XR']):
        pBSs = []
        for mode in modes:
            BSs = self._compute_BS(startfrom = mode)
            for x, r in BSs:
                if self.check_no_ecqs(x, r):
                    idx, det = self.compute_idx_and_det(x, r)
                    if idx == 0:
                        pBSs.append( (x, r, idx, det,) )
        order = np.argsort([len(bs[0]) for bs in pBSs])
        pBSs   = [pBSs[order[id]] for id in range(len(order))]
        return pBSs[0]

def indicator_subset(crn, modes = ['R', 'X', 'XR']):
    """ I dentify the indicator subset of a Chemical Reaction Network. """
    assert isinstance(crn, CRN)
    flag = True
    res  = []
    mBS  = crn.proper_mBS(modes)
    while True:
        res.append({"chemicals": [crn._X_names[m] for m in mBS[0]],
                    "reactions": [crn._R_names[n] for n in mBS[1]],
                    "index": mBS[2],
                    "sign_det":  mBS[3]})
        if (len(mBS[0]) == len(crn._X_names)):
            break
        else:
            crn.structural_reduction(mBS[0], mBS[1])
            mBS  = crn.proper_mBS()
    return res

# ==== [ END OF THE SCRIPT ] ================================================================================ #
