# cython: language_level = 3
# cython: language = c++
# cython: cdivision = True
# cython: wraparound = False
# cython: boundscheck = False

cimport cython
from libc.stdlib cimport malloc, free
from cython.parallel  import prange
cimport numpy as cnp
import numpy as np
from libcpp cimport bool as bool_t
from scipy.linalg.cython_lapack cimport dgetrf
from scipy.optimize import minimize

cdef extern from "<math.h>" nogil:
    const double INFINITY
    const double NAN
    double fabs(double z)
    int remainderl(int x, int y)
    double atan(double x)

cdef double INF = INFINITY
cdef double TOL = 1e-8

cdef bool_t iszero(double z) nogil:
    return fabs(z) < TOL

cdef int cyswaprows(double[:,:] MAT, int cols, int i, int j) nogil:
    """ Swap rows i and j in matrix mat. """
    cdef double* temp = <double*> malloc(cols * sizeof(double))
    cdef int k
    for k in prange(cols):
        temp[k] = MAT[i, k]
        MAT[i, k] = MAT[j, k]
        MAT[j, k] = temp[k]
    free(temp)
    return 1

cdef int cyrref(double[:,:] MAT, int rows, int cols) nogil:
    """ Compute the reduced row echelon form of a given matrix. """
    cdef int row = 0, col = 0, i, j, countdown, pivot_row
    cdef double pivot
    for countdown in range(rows + cols):
        if (col >= cols) or (row >= rows):
            break
        pivot_row = row
        for i in range(row + 1, rows):
            if fabs(MAT[i, col]) > fabs(MAT[pivot_row, col]):
                pivot_row = i
        if iszero(MAT[pivot_row, col]):
            col += 1
            continue
        # Swap rows to move pivot row to the current row
        if pivot_row != row:
            cyswaprows(MAT, cols, row, pivot_row)
        # Normalize the pivot row
        pivot = MAT[row, col]
        for j in prange(cols):
            MAT[row, j] = (MAT[row, j] / pivot)
        # Eliminate entries above and below the pivot (reduction)
        for i in range(rows):
            if i != row:
                pivot = MAT[i, col]
                for j in prange(cols):
                    MAT[i, j] -= (MAT[row, j] * pivot)
        row += 1
        col += 1
    return 1

cdef int cykerimg(double[:,:] mat, double[:,:] WORK, int rows, int cols) nogil except -1:
    """
    Using gaussian elimination to compute the kernel and the image.
    The return value is the nullity (dimension of kernel space).
    """
    cdef int row, col
    WORK[...] = 0.0
    for col in prange(cols):
        for row in prange(rows):
            WORK[col, row] = mat[row, col]
        WORK[col, rows+col] = 1.0
    cyrref(WORK, cols, cols+rows)
    for col in range(cols):
        for row in range(rows):
            if not iszero(WORK[cols-col-1, row]):
                return col
    return cols

cpdef cnp.ndarray[double, ndim=2] ker(cnp.ndarray[double, ndim=2] mat):
    """ Compute the (right-)null space of a given matrix. """
    cdef int rows = mat.shape[0], cols = mat.shape[1], nullity
    cdef cnp.ndarray[double, ndim=2] WORK
    WORK = np.zeros(shape = (cols, cols+rows), dtype = float)
    nullity = cykerimg(mat, WORK, rows, cols)
    return WORK[cols-nullity:,rows:].T

cpdef cnp.ndarray[double, ndim=2] construct_A_from(cnp.ndarray[double, ndim=2] stoi,
                                                   cnp.ndarray[double, ndim=2] reginfo):
    """ Construct the A matrix from stoichiometric and regulation information. """
    cdef cnp.ndarray[double, ndim=2] C, DT
    cdef int M, N
    C = ker(stoi)
    DT = ker(stoi.T).T
    M = stoi.shape[0]
    N = stoi.shape[1]
    A = np.zeros((M + C.shape[1], N + DT.shape[0]))
    A[:N, :M] = reginfo.T
    A[:N, M:] = C
    A[N:, :M] = DT
    return A

cdef double cydet(double[:,:] MAT, int n, int[:] IPIV, int INFO) nogil except NAN:
    """ Compute the determinant of a square matrix MAT using DGETRF from the LAPACK library. """
    cdef int j
    cdef double detval = 1.0
    dgetrf(&n, &n, &MAT[0,0], &n, &IPIV[0], &INFO)
    for j in range(n):
        if j != (IPIV[j] - 1):
            detval *= -(MAT[j, j])
        else:
            detval *= MAT[j, j]
    return detval

def signdet(A, trials = 10):
    varposition = ~(np.abs(A) < INF)
    varlength   = np.sum(varposition)
    if varlength == 0:
        return np.sign(cydet(A.copy(), A.shape[0], np.zeros(A.shape[0], dtype=np.int32), 1))
    Pposits  = varposition * (np.sign(A) > 0)
    Nposits  = varposition * (np.sign(A) < 0)
    def vardetA(varsz):
        varA = A.copy()
        varA[varposition] = varsz
        varA[Pposits] =  varA[Pposits] ** 2
        varA[Nposits] = -varA[Nposits] ** 2
        detval = cydet(varA, varA.shape[0], np.zeros(varA.shape[0], dtype=np.int32), 1)
        return atan(detval)
    Pattempt = []
    Nattempt = []
    for _t in range(trials):
        Pattempt.append(-minimize(fun = lambda z: -vardetA(z),
                                  x0 = np.random.normal(size = varlength, scale = 10)).fun >  TOL)
        Nattempt.append( minimize(fun = vardetA,
                                  x0 = np.random.normal(size = varlength, scale = 10)).fun < -TOL)
    has_positive = np.any(Pattempt)
    has_negative = np.any(Nattempt)
    if has_positive and has_negative:
        return  NAN
    elif has_positive:
        return  INF
    elif has_negative:
        return -INF
    else:
        return  0.0

cdef int cyadjugate(double[:,:] MAT, int n, double[:,:,:] WORK, int[:,:] IPIV) nogil except 0:
    """ Compute the adjugate matrix of a square matrix mat. """
    cdef int i, j, row, col, nm1 = n-1
    for row in prange(n):
        for col in prange(n):
            for i in prange(nm1):
                for j in prange(nm1):
                    if (i>=row) and (j>=col):
                        WORK[n*row+col,i,j] = MAT[i+1,j+1]
                    elif j >= col:
                        WORK[n*row+col,i,j] = MAT[i,j+1]
                    elif i >= row:
                        WORK[n*row+col,i,j] = MAT[i+1,j]
                    else:
                        WORK[n*row+col,i,j] = MAT[i,j]
    for row in prange(n):
        for col in prange(n):
            if (cython.cmod(row+col, 2) == 1):
                MAT[col, row] = -cydet(WORK[n*row+col], n-1, IPIV[n*row+col], 0)
            else:
                MAT[col, row] =  cydet(WORK[n*row+col], n-1, IPIV[n*row+col], 0)
    return 1

cpdef adj(double[:,:] A):
    """ Compute the adjugate matrix of a square matrix. """
    n = A.shape[0]
    WORK = np.empty(shape = (n*n, n-1, n-1), dtype = float)
    IPIV = np.zeros(shape = (n*n, n-1), dtype = np.int32)
    cyadjugate(A, n, WORK, IPIV)
    return np.array(A)

cpdef cnp.ndarray[bool_t, ndim = 2] ecqavoider(cnp.ndarray[double, ndim=2] DT):
    cdef int L = DT.shape[0], M = DT.shape[1]
    cdef cnp.ndarray[bool_t, ndim=2] DTbool = (np.abs(DT) > TOL)
    cdef cnp.ndarray[bool_t, ndim=2] MAT = np.dot(DTbool.T, DTbool)
    cdef int l
    for l in range(L):
        MAT = np.dot(MAT, MAT)
    for m in range(M):
        MAT[m,m] = True
    return MAT

cdef bool_t[:] cy_iBSsearcher(bool_t[:] X, bool_t[:] R,
                              bool_t[:,:] _ecq_avoider,
                              bool_t[:,:] _oc_keeper,
                              bool_t[:,:] _lol_keeper,
                              int M, int N):
    cdef bool_t[:] newX = np.zeros(shape = (M,), dtype = np.bool_) 
    cdef bool_t[:] newR = np.zeros(shape = (N,), dtype = np.bool_)
    cdef bool_t[:] gamma = np.zeros(shape = (M+N,), dtype = np.bool_)
    cdef int p = 0, P = M+N, q
    cdef bool_t flag_updated
    newX[...] = X
    newR[...] = R
    for p in range(P):
        newX += np.dot(_ecq_avoider, newX)
        newR += np.dot(_oc_keeper, newX)
        newX += np.dot(_lol_keeper, newR)
        flag_updated = False
        for q in prange(M, nogil = True):
            if newX[q] != X[q]:
                flag_updated = True
        for q in prange(N, nogil = True):
            if newR[q] != R[q]:
                flag_updated = True
        if not flag_updated:
            break
    for p in prange(M, nogil=True):
        gamma[p] = newX[p]
    for p in prange(N, nogil=True):
        gamma[M+p] = newR[p]
    return gamma

cdef int cy_chiindex(cnp.ndarray[double, ndim=2] stoi,
                     cnp.ndarray[double, ndim=2] DT,
                     bool_t[:] Xg, bool_t[:] Rg):
    cdef cnp.ndarray[double, ndim=2] C11
    cdef int idx, nRg = np.sum(Rg), nXg = np.sum(Xg), cokers, cycles
    C11 = ker(stoi[Xg,:][:,Rg])
    cokers = nXg -ker(DT[:,Xg]).shape[1]
    cycles = C11.shape[1] if (nXg != 0) else 0
    idx = nRg -nXg +cokers -cycles
    return idx

class CRN:
    """ Chemical Reaction Network. """
    def __init__(self, stoi, reginfo = None, X_names = None, R_names = None):
        self.stoi    = stoi
        if reginfo is None:
            reginfo = np.zeros(shape = stoi.shape, dtype = float)
            reginfo[stoi < -TOL] = np.inf
            self.reginfo = reginfo
        else:
            self.reginfo = reginfo
        if X_names is None:
            self.X_names = ['X{:d}'.format(m) for m in range(self.stoi.shape[0])]
        else:
            self.X_names = X_names
        if R_names is None:
            self.R_names = ['R{:d}'.format(n) for n in range(self.stoi.shape[1])]
        else:
            self.R_names = R_names
        self.A = construct_A_from(stoi, reginfo)
        self._ecq_avoider = np.array(ecqavoider(self.A[self.stoi.shape[1]:,:self.stoi.shape[0]]), dtype = np.bool_)
        self._oc_keeper   = ~(np.abs(self.reginfo) < TOL)

    def _sensitivity_(self, trials = 1):
        boolS = np.zeros(shape = self.stoi.shape, dtype = np.bool_)
        varpositions = ~(np.abs(self.A) < INF)
        NL = self.A.shape[0]
        WORK = np.empty(shape = (NL*NL, NL-1, NL-1), dtype = float)
        IPIV = np.zeros(shape = (NL*NL, NL-1), dtype = np.int32)
        M, N = self.stoi.shape
        num_vars = np.sum(varpositions)
        for n in range(trials):
            randA = self.A.copy()
            randA[varpositions] = np.random.normal(size = num_vars)
            cyadjugate(randA, NL, WORK, IPIV)
            boolS = boolS + (np.abs(randA) > TOL)[:M,:N]
        return boolS

    def iBSsearcher(self, X, R):
        """ Start from a subnetwork, searching for a regulatory module w/o eCQs. """
        # Not really used but just for convenience.
        M, N = self.stoi.shape
        if not isinstance(X[0].dtype, np.bool_):
            X = np.array([(m in X) for m in range(M)], dtype = np.bool_)
        if not isinstance(R[0].dtype, np.bool_):
            R = np.array([(n in R) for n in range(N)], dtype = np.bool_)
        S = self._sensitivity_()
        res = cy_iBSsearcher(X, R, self._ecq_avoider, self._oc_keeper.T, S, M, N)
        res = np.array(res)
        return res[:M], res[M:]

    def iBSs(self):
        """ Searching for regulatory modules w/o eCQs. """
        M, N = self.stoi.shape
        X = np.zeros(shape=(M,), dtype = np.bool_)
        R = np.zeros(shape=(N,), dtype = np.bool_)
        S = self._sensitivity_()
        DT = self.A[self.stoi.shape[1]:,:self.stoi.shape[0]]
        iBSs_list = []
        for m in range(-1, M):
            for n in range(N):
                X[::1] = False
                if (m >= 0):
                    X[m] = True
                R[::1] = False
                R[n] = True
                res  = cy_iBSsearcher(X, R, self._ecq_avoider, self._oc_keeper.T, S, M, N)
                res  = np.array(res)
                X, R = res[:M], res[M:]
                index = cy_chiindex(self.stoi, DT, X, R)
                if index == 0:
                    iBSs_list.append( (X.copy().tolist(), R.copy().tolist(), index,) )
        if len(iBSs_list) != 0:
            # remove the duplicates
            uniqueiBSs = []
            L = len(iBSs_list)
            for a in range(L):
                if np.all([(iBSs_list[a] != iBSs_list[b]) for b in range(a+1, L)]):
                    uniqueiBSs.append( iBSs_list[a] )
            L = len(uniqueiBSs)
            # and bubble sort the subnetworks
            for a in range(L-1):
                for b in range(a+1, L-1):
                    if np.sum(uniqueiBSs[a][0]) > np.sum(uniqueiBSs[b][0]):
                        temp = uniqueiBSs[a]
                        uniqueiBSs[a] = uniqueiBSs[b]
                        uniqueiBSs[b] = temp
                    elif np.sum(uniqueiBSs[a][0]) == np.sum(uniqueiBSs[b][0]):
                        if np.sum(uniqueiBSs[a][1]) > np.sum(uniqueiBSs[b][1]):
                            temp = uniqueiBSs[a]
                            uniqueiBSs[a] = uniqueiBSs[b]
                            uniqueiBSs[b] = temp
                    else:
                        pass
        else: # fail to find any subnetworks for the structural reduction
            X[:] = True
            R[:] = True
            index = cy_chiindex(self.stoi, DT, X, R)
            uniqueiBSs = [ (X.copy().tolist(), R.copy().tolist(), index,) ]
        return uniqueiBSs

    def det_subnetwork(self, X, R, struc_reduc = False, evaluate = True):
        M, N = self.stoi.shape
        X = np.array(X, dtype = np.bool_)
        R = np.array(R, dtype = np.bool_)
        if evaluate:
            NU11 = self.stoi[X,:][:,R]
            pRpX = self.reginfo[X,:][:,R]
            Agamma = np.array(construct_A_from(NU11, pRpX))
            signdetAg = signdet(Agamma)
        else:
            signdetAg = None
        if struc_reduc and (np.sum(X) < self.stoi.shape[0]):
            # Start updating
            NU = self.stoi
            nured = NU[~X,:][:,~R] - (NU[~X,:][:,R]).dot(np.linalg.pinv(NU[X,:][:,R])).dot(NU[X,:][:,~R])
            self.stoi = nured
            self.reginfo = self.reginfo[~X,:][:,~R]
            self.A = construct_A_from(self.stoi, self.reginfo)
            self._ecq_avoider = np.array(self._ecq_avoider)[~X,:][:,~X]
            self._oc_keeper   = ~(np.abs(self.reginfo) < TOL)
            self.X_names = [self.X_names[m] for m in range(M) if not X[m]]
            self.R_names = [self.R_names[n] for n in range(N) if not R[n]]
        return signdetAg


def indicator_diagnose(crn):
    assert isinstance(crn, CRN)
    res = []
    N   = crn.stoi.shape[1]
    for n in range(N):
        ibs = crn.iBSs()
        mBS = ibs[0]
        X = [crn.X_names[m] for m in range(len(crn.X_names)) if mBS[0][m]]
        R = [crn.R_names[n] for n in range(len(crn.R_names)) if mBS[1][n]]
        signdet = crn.det_subnetwork(mBS[0], mBS[1], struc_reduc = True)
        res.append( {'chemicals': X, 'reactions': R, 'index': mBS[2], 'sign_det': signdet} )
        if len(X) == crn.stoi.shape[0]:
            break
    return res

# ===========================
