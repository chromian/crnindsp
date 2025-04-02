# cython: language_level=3
# cython: language=c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# =================================================================================================================== #
cimport cython
cimport numpy as cnp
import  numpy as  np
from libc.stdlib          cimport malloc, free
from cython.parallel      cimport prange
from libcpp               cimport bool as bool_t
from scipy.optimize        import minimize
from scipy.linalg.cython_lapack cimport dgetrf
# =================================================================================================================== #

cdef double TOL = 1e-8  # Tolerance for zero comparison

# External C functions
cdef extern from "<math.h>" nogil:
    const double INFINITY
    const double NAN
    double fabs(double x)
    double atan(double x)

cdef inline bool_t iszero(double z) nogil:
    """Check if a value is within tolerance of zero."""
    return fabs(z) < TOL

cdef void cy_rref(double[:, :] MAT, int rows, int cols) nogil:
    """
    Compute the reduced row echelon form (RREF) of a matrix in-place.

    Parameters:
    -----------
    MAT : memoryview
        Input/output matrix to be transformed into RREF
    rows : int
        Number of rows in the matrix
    cols : int
        Number of columns in the matrix
    """
    cdef:
        int row = 0, col = 0
        int i, j, countdown, pivot_row
        double pivot
        double* temp = <double*> malloc(cols * sizeof(double))

    if temp == NULL:
        return  # Memory allocation failed

    try:
        # Process each column or row until exhausted
        for countdown in range(rows + cols):
            if col >= cols or row >= rows:
                break
            # Find pivot row with maximum absolute value
            pivot_row = row
            for i in range(row + 1, rows):
                if fabs(MAT[i, col]) > fabs(MAT[pivot_row, col]):
                    pivot_row = i
            # Skip column if pivot is effectively zero
            if iszero(MAT[pivot_row, col]):
                col += 1
                continue
            # Swap rows if pivot isn't in current row
            if pivot_row != row:
                for i in prange(cols, nogil=True):
                    temp[i] = MAT[row, i]
                    MAT[row, i] = MAT[pivot_row, i]
                    MAT[pivot_row, i] = temp[i]
            # Normalize pivot row
            pivot = MAT[row, col]
            for j in prange(cols, nogil=True):
                MAT[row, j] /= pivot
            # Eliminate column entries above and below pivot
            for i in range(rows):
                if i != row:
                    pivot = MAT[i, col]
                    for j in prange(cols, nogil=True):
                        MAT[i, j] -= MAT[row, j] * pivot
            row += 1
            col += 1
    finally:
        free(temp)

cdef int cy_nullity(double[:, :] mat, double[:, :] WORK, int rows, int cols) nogil except -1:
    """
    Compute the nullity (dimension of kernel space) using Gaussian elimination.

    Parameters:
    -----------
    mat : memoryview
        Input matrix
    WORK : memoryview
        Working matrix (cols x (rows + cols)), modified in-place
    rows : int
        Number of rows in the input matrix
    cols : int
        Number of columns in the input matrix

    Returns:
    --------
    int
        Nullity of the matrix (number of free variables)
    """
    cdef int row, col

    # Initialize working matrix: transpose mat and augment with identity
    WORK[...] = 0.0
    for col in prange(cols, nogil=True):
        for row in range(rows):
            WORK[col, row] = mat[row, col]
        WORK[col, rows + col] = 1.0
    # Compute RREF of augmented matrix
    cy_rref(WORK, cols, cols + rows)
    # Count zero rows from right to determine nullity
    for col in range(cols):
        for row in range(rows):
            if not iszero(WORK[cols - col - 1, row]):
                return col
    return cols

cdef double cy_det(double[:, :] MAT, int n) nogil except NAN:
    """
    Compute the determinant of a square matrix using LAPACK's DGETRF.

    Parameters:
    -----------
    MAT : memoryview
        Input square matrix (n x n), modified in-place by LU factorization
    n : int
        Dimension of the square matrix

    Returns:
    --------
    double
        Determinant of the matrix

    Raises:
    -------
    ValueError
        If memory allocation fails or LAPACK's DGETRF returns an error
    """
    if n <= 0:
        return 0.0
    cdef:
        int j, INFO = 0
        int* IPIV = <int*> malloc(n * sizeof(int))
        double detval = 1.0

    if IPIV == NULL:
        return NAN

    try:
        # Perform LU factorization with partial pivoting
        dgetrf(&n, &n, &MAT[0, 0], &n, IPIV, &INFO)
        # Check for LAPACK error
        if INFO != 0:
            return NAN
        # Compute determinant with sign adjustments for pivots
        for j in range(n):
            if j != (IPIV[j] - 1):  # Row swap occurred
                detval *= -MAT[j, j]
            else:
                detval *= MAT[j, j]
        return detval
    finally:
        free(IPIV)

cdef double[:,:] cy_adjugate(double[:, :] MAT, int n, double[:, :, :] WORK) nogil:
    """
    Compute the adjugate (classical adjoint) matrix of a square matrix in-place.

    Parameters:
    -----------
    MAT : memoryview
        Input/output square matrix (n x n), overwritten with its adjugate
    n : int
        Dimension of the square matrix
    WORK : memoryview
        3D working array (n x n x (n-1) x (n-1)) for minor matrices

    Returns:
    -------
    double[:, :]
        Adjugate matrix (n x n)

    Raises:
    -------
    ValueError
        If determinant computation fails in cy_det
    """
    cdef:
        int i, j, row, col
        int nm1 = n - 1

    # Populate WORK with minor matrices
    for row in prange(n, nogil=True):
        for col in prange(n):
            for i in prange(nm1):
                for j in prange(nm1):
                    if i >= row and j >= col:
                        WORK[n * row + col, i, j] = MAT[i + 1, j + 1]
                    elif j >= col:
                        WORK[n * row + col, i, j] = MAT[i, j + 1]
                    elif i >= row:
                        WORK[n * row + col, i, j] = MAT[i + 1, j]
                    else:
                        WORK[n * row + col, i, j] = MAT[i, j]

    # Compute cofactors and transpose to form adjugate
    for row in prange(n, nogil=True):
        for col in prange(n):
            if cython.cmod(row + col, 2) == 1:
                MAT[col, row] = -cy_det(WORK[n * row + col], nm1)
            else:
                MAT[col, row] =  cy_det(WORK[n * row + col], nm1)

    return MAT

cdef double[:, :] cy_construct_A_from(cnp.ndarray[double, ndim=2] stoi, double[:, :] reginfo):
    """
    Construct the A matrix from stoichiometry and regulation information.

    Parameters:
    -----------
    stoi : ndarray
        Stoichiometric matrix (num_chem x num_react)
    reginfo : memoryview
        Regulation information matrix (num_chem x num_react)

    Returns:
    --------
    double[:, :]
        Constructed the A-matrix (Adim x Adim), where Adim = num_chem + nullity(stoi)
    """
    cdef:
        double[:, :] WORK, Amat
        int null, conull, i, j
        int num_chem = stoi.shape[0]
        int num_react = stoi.shape[1]
        int Adim

    # Step 1: Compute nullity of stoichiometry matrix
    WORK = np.zeros(shape=(num_react, num_react + num_chem), dtype=float)
    null = cy_nullity(stoi, WORK, num_chem, num_react)
    Adim = num_chem + null
    # Initialize output matrix
    Amat = np.zeros(shape=(Adim, Adim), dtype=float)
    # Step 2: Fill Amat with null space basis from stoichiometry
    for i in prange(num_react, nogil=True):
        for j in prange(null):
            Amat[i, num_chem + j] = WORK[num_react - null + j, num_chem + i]
    # Step 3: Compute co-nullity (nullity of transpose) and fill remaining parts
    WORK = np.zeros(shape=(num_chem, num_react + num_chem), dtype=float)
    conull = cy_nullity(stoi.T, WORK, num_react, num_chem)
    for j in prange(num_chem, nogil=True):
        for i in prange(conull):
            Amat[num_react + i, j] = WORK[num_chem - conull + i, num_react + j]
        for i in prange(num_react):
            Amat[i, j] = reginfo[j, i]

    return Amat

def signdet(A, trials=10):
    """
    Determine the sign of the determinant of matrix A with variable entries.

    Parameters:
    -----------
    A : ndarray
        Input square matrix with possible infinite or variable entries
    trials : int, optional
        Number of optimization trials to test sign consistency (default: 10)

    Returns:
    --------
    float
        Sign of the determinant:
        =  1.0 if consistently positive
        = -1.0 if consistently negative
        =  0.0 if consistently zero
        =  NAN if both positive and negative signs are possible
    """
    # Identify variable positions (non-infinite entries)
    varposition = ~(np.abs(A) < INFINITY)
    varlength = np.sum(varposition)
    # Case 1: No variable entries, compute sign directly
    if varlength == 0:
        detval = np.linalg.det(A.copy())
        if abs(detval) < TOL:
            return 0
        else:
            return np.sign(detval)
    # Define positions for positive and negative variables
    Pposits = varposition & (np.sign(A) > 0)
    Nposits = varposition & (np.sign(A) < 0)
    # Objective function for optimization
    def vardetA(varsz):
        """Compute arctan of determinant with squared variables."""
        varA = np.array(A).copy()
        varA[varposition] = varsz
        varA[Pposits] = varA[Pposits] ** 2   # Ensure positive
        varA[Nposits] = -(varA[Nposits] ** 2)  # Ensure negative
        detval = cy_det(varA, varA.shape[0])
        return atan(detval)
    # Run optimization trials to test sign consistency
    Pattempt = []
    Nattempt = []
    for _ in range(trials):
        # Maximize det (minimize -vardetA) for positive sign
        Presult = minimize(fun=lambda z: -vardetA(z),
                           x0=np.random.normal(size=varlength, scale=10.0))
        Pattempt.append(-Presult.fun > TOL)  # True if det > 0

        # Minimize det for negative sign
        Nresult = minimize(fun=vardetA,
                           x0=np.random.normal(size=varlength, scale=10.0))
        Nattempt.append(Nresult.fun < -TOL)  # True if det < 0

    # Determine sign based on trial outcomes
    has_positive = np.any(Pattempt)
    has_negative = np.any(Nattempt)

    if has_positive and has_negative:
        return  NAN
    elif has_positive:
        return  1.0
    elif has_negative:
        return -1.0
    else:
        return  0.0

cdef bool_t[:, :] cy_cq_classifier(cnp.ndarray[double, ndim=2] coker_basis):
    """
    Classify chemicals by the equivalence relation based on cokernel basis.

    Parameters:
    -----------
    coker_basis : ndarray
        Cokernel basis matrix (num_cqs x num_chem)

    Returns:
    --------
    bool_t[:, :]
        Transitive closure matrix (num_chem x num_chem) indicating the equivalence class.
    """
    cdef:
        int num_chem = coker_basis.shape[1]
        int num_cqs = coker_basis.shape[0]
        int i, j, k, l = 0
        bool_t[:, :] WORK = np.zeros(shape=(num_chem, num_chem), dtype=np.bool_)

    # Step 1: Identify direct relationships from coker_basis
    for i in prange(num_chem, nogil=True):
        for j in prange(num_chem):
            for k in prange(num_cqs):
                if not (iszero(coker_basis[k, i]) | iszero(coker_basis[k, j])):
                    WORK[i, j] = True

    # Step 2: Compute transitive closure
    for l in range(num_cqs):
        for i in prange(num_chem, nogil=True):
            for j in prange(num_chem):
                for k in prange(num_chem):
                    if WORK[i, k] and WORK[k, j]:
                        WORK[i, j] = True

    # Step 3: Set diagonal to True (reflexive relation)
    for i in prange(num_chem, nogil=True):
        WORK[i, i] = True

    return WORK

cdef bool_t[:] cy_isoBS_searcher(bool_t[:] XandR,
                                 bool_t[:, :] oc_keeper,
                                 bool_t[:, :] lo_keeper,
                                 bool_t[:, :] cq_class,
                                 int num_chem, int num_react) nogil:
    """
    Search for an isolated buffered subnetwork by expanding XandR.

    Parameters:
    -----------
    XandR : memoryview
        Input/output boolean array (num_chem + num_react) indicating species and reactions
    oc_keeper : memoryview
        Boolean matrix (num_chem x num_react) for output-completeness
    lo_keeper : memoryview
        Boolean matrix (num_chem x num_react) for localization
    cq_class : memoryview
        Boolean matrix (num_chem x num_chem) for the equivalence classes based on cokernel
    num_chem : int
        Number of chemical species
    num_react : int
        Number of reactions

    Returns:
    --------
    bool_t[:]
        Updated XandR array representing the isolated buffered subnetwork.
        Note that the output is not necessarily isolated buffering structures, but candicates.
    """
    cdef:
        int m, n, p
        int P = num_chem + num_react
        bool_t* newXR = <bool_t*> malloc(P * sizeof(bool_t))
        bool_t flag_updated

    try:
        for p in prange(P):
            newXR[p] = XandR[p]
        for p in prange(P, nogil = False):
            # make the subnetwork output-complete
            for m in prange(num_chem, nogil = True):
                if newXR[m]:
                    for n in prange(num_react):
                        if oc_keeper[m, n]:
                            newXR[num_chem + n] = True
            # let the subnetwork localize the responses to perturbation
            for n in prange(num_react, nogil = True):
                if newXR[num_chem + n]:
                    for m in prange(num_chem):
                        if lo_keeper[m, n]:
                            newXR[m] = True
            # let the subnetwork has not emergenct CQs
            for m in prange(num_chem, nogil = True):
                if newXR[m]:
                    for n in prange(num_chem):
                        if cq_class[m, n]:
                            newXR[n] = True
            # check whether the updation stops
            flag_updated = False
            for m in prange(P):
                if newXR[p] != XandR[p]:
                    flag_updated = True
            if not flag_updated:
                break
            else:
                for m in prange(P):
                    XandR[p] = newXR[p]
        # save the result to the original array
        for p in prange(P):
            XandR[p] = newXR[p]

    finally:
        free(newXR)

    return XandR

cdef bool_t cy_no_emergent_cycles(cnp.ndarray[double, ndim=2] stoi, bool_t[:] XandR):
    """
    Check if a subnetwork has no emergent cycles based on its nullity and orthogonality.

    Parameters:
    -----------
    stoi : ndarray
        Stoichiometry matrix (M x N)
    XandR : memoryview
        Boolean array (M + N) indicating selected species and reactions

    Returns:
    --------
    bool_t
        True if no emergent cycles exist, False otherwise

    Raises:
    -------
    ValueError
        If nullity computation fails
    """
    cdef:
        int M = stoi.shape[0]         # Number of chemical species
        int N = stoi.shape[1]         # Number of reactions
        int M1, N1                    # Number of selected species and reactions
        int nullity11, m, n
        cnp.ndarray[bool_t, ndim=1] X = np.zeros(shape=(M,), dtype=np.bool_)
        cnp.ndarray[bool_t, ndim=1] R = np.zeros(shape=(N,), dtype=np.bool_)
        double[:, :] WORK             # Working array for nullity
        double[:, :] C11              # Null space basis of subnetwork
        double[:, :] nu01C11          # Orthogonality check matrix

    M1 = np.sum(XandR[:M])
    N1 = np.sum(XandR[M:])
    # Extract selected species and reactions
    for m in prange(M, nogil = True):
        X[m] = XandR[m]
    for n in prange(N, nogil = True):
        R[n] = XandR[M + n]
    # Compute nullity of the subnetwork stoichiometry
    WORK = np.zeros(shape=(N1, N1 + M1), dtype=float)
    nullity11 = cy_nullity(stoi[X, :][:, R], WORK, M1, N1)
    # Extract null space basis (C11)
    C11 = WORK[N1 - nullity11:, M1:]
    # Check orthogonality with non-selected species
    nu01C11 = np.dot(stoi[~X, :][:, R], C11.T)
    # Test if all elements are within tolerance
    for m in range(nu01C11.shape[0]):
        for n in range(nu01C11.shape[1]):
            if fabs(nu01C11[m, n]) >= TOL:
                return False
    return True


cdef int cy_chiindex(cnp.ndarray[double, ndim=2] stoi,
                     cnp.ndarray[double, ndim=2] coker_basis,
                     bool_t[:] XandR):
    """
    Compute the chi-index of a chemical reaction subnetwork.

    Parameters:
    -----------
    stoi : ndarray
        Stoichiometry matrix (M x N)
    coker_basis : ndarray
        Cokernel basis matrix (L x M)
    XandR : memoryview
        Boolean array (M + N) indicating selected species and reactions

    Returns:
    --------
    int
        Chi-index of the subnetwork (-#chemicals + #reactions - #cycles + #CQs)
    """
    cdef:
        int M = stoi.shape[0]          # Number of chemical species
        int N = stoi.shape[1]          # Number of reactions
        int L = coker_basis.shape[0]   # Number of cokernel basis vectors
        int M1 = 0                     # Number of selected species
        int N1 = 0                     # Number of selected reactions
        int num_CQs, num_cycles, chi
        double[:, :] WORK

    # Count selected species (M1) and reactions (N1)
    for i in range(M):
        if XandR[i]:
            M1 += 1
    for i in range(N):
        if XandR[M + i]:
            N1 += 1
    # Compute number of cycles in the subnetwork
    WORK = np.zeros(shape=(N1, M + N1), dtype=float)
    num_cycles = cy_nullity(stoi[:, XandR[M:]], WORK, M, N1)
    # Compute number of conserved quantities (CQs)
    WORK = np.zeros(shape=(M1, M1 + L), dtype=float)
    num_CQs = M1 - cy_nullity(coker_basis[:, XandR[:M]], WORK, L, M1)
    # Calculate chi-index
    chi = -M1 + N1 - num_cycles + num_CQs
    return chi


class CRN:
    """Chemical Reaction Network (CRN) representation and analysis."""

    def __init__(self, stoi, reginfo=None, X_names=None, R_names=None, name=None):
        """
        Initialize a Chemical Reaction Network.

        Parameters:
        -----------
        stoi : ndarray
            Stoichiometry matrix (num_chem x num_react)
        reginfo : ndarray, optional
            Regulation information matrix (num_chem x num_react), defaults to infinity where stoi < -TOL
        X_names : list of str, optional
            Names of chemical species, defaults to ['X0', 'X1', ...]
        R_names : list of str, optional
            Names of reactions, defaults to ['R0', 'R1', ...]
        name : str, optional
            Name of the CRN
        """
        self.name = name
        self.stoi = np.asarray(stoi, dtype=float)

        # Set regulation info
        if reginfo is None:
            reginfo = self.stoi.copy() * 0.0
            reginfo[self.stoi < -TOL] = INFINITY
        self.reginfo = np.asarray(reginfo, dtype=float)

        # Set chemical species names
        self.X_names = [f'X{m}' for m in range(self.stoi.shape[0])] if X_names is None else X_names

        # Set reaction names
        self.R_names = [f'R{n}' for n in range(self.stoi.shape[1])] if R_names is None else R_names

        # Compute A matrix and derived properties
        self.A = cy_construct_A_from(self.stoi, self.reginfo)
        self.__cq_class = cy_cq_classifier(np.array(self.A[self.stoi.shape[1]:, :self.stoi.shape[0]]))
        self.__oc_keeper = ~(np.abs(self.reginfo) < TOL)

    def _smat_(self, trial=2):
        """
        Compute a boolean sensitivity matrix using random perturbations.

        Parameters:
        -----------
        trial : int, optional
            Number of random trials to aggregate (default: 2)

        Returns:
        --------
        ndarray
            Boolean sensitivity matrix (num_chem x num_react)
        """
        cdef:
            int M = self.stoi.shape[0]
            int N = self.stoi.shape[1]
            int L = self.A.shape[0]
            cnp.ndarray[bool_t, ndim=2] booleanS = np.zeros(shape=(M, N), dtype=np.bool_)
            cnp.ndarray[double, ndim=2] randA
            cnp.ndarray[bool_t, ndim=2] varholder = ~(np.abs(self.A) < INFINITY)
            int _t_
            double[:, :, :] WORK = np.empty(shape=(L * L, L - 1, L - 1), dtype=float)

        for _t_ in range(trial):
            randA = np.array(self.A.copy())
            randA[varholder] = np.random.normal(size=np.sum(varholder), scale=10.0)
            adjA = cy_adjugate(randA, L, WORK)
            booleanS |= (np.abs(adjA) > TOL)[:M, :N]  # In-place OR for aggregation

        return booleanS

    def isoBS_searchfrom(self, X, R, trial=2):
        """
        Search for an isolated buffered subnetwork starting from given species and reactions.

        Parameters:
        -----------
        X : list of int
            Indices of initial chemical species
        R : list of int
            Indices of initial reactions
        trial : int, optional
            Number of trials for sensitivity matrix (default: 2)

        Returns:
        --------
        tuple
            (Xs, Rs) - Lists of species and reaction indices in the subnetwork
        """
        cdef:
            int m, n, M = self.stoi.shape[0], N = self.stoi.shape[1]
            bool_t[:] XandR = np.zeros(shape=(M + N), dtype=np.bool_)
            bool_t[:, :] lo_keeper

        # Initialize XandR from X and R
        for m in range(M):
            XandR[m] = m in X
        for n in range(N):
            XandR[M + n] = n in R

        # Compute sensitivity matrix and search for subnetwork
        lo_keeper = self._smat_(trial)
        XandR = cy_isoBS_searcher(XandR, self.__oc_keeper, lo_keeper, self.__cq_class, M, N)

        # Extract resulting species and reactions
        Xs = [m for m in range(M) if XandR[m]]
        Rs = [n for n in range(N) if XandR[M + n]]
        return Xs, Rs

    def compute_index(self, X, R):
        """
        Compute the chi-index for a subnetwork defined by species and reactions.

        Parameters:
        -----------
        X : list of int
            Indices of chemical species
        R : list of int
            Indices of reactions

        Returns:
        --------
        int
            Chi-index of the subnetwork
        """
        cdef:
            int m, n, M = self.stoi.shape[0], N = self.stoi.shape[1]
            cnp.ndarray[double, ndim=2] DT
            bool_t[:] XandR = np.zeros(shape=(M + N), dtype=np.bool_)

        # Initialize XandR from X and R
        for m in range(M):
            XandR[m] = m in X
        for n in range(N):
            XandR[M + n] = n in R

        # Compute chi-index using the cokernel part of A
        DT = np.array(self.A[N:, :M])
        return cy_chiindex(self.stoi, DT, XandR)

    def search_for_isoBSs(self, trial=2):
        """
        Search for all unique isolated buffered subnetworks with chi-index 0.

        Parameters:
        -----------
        trial : int, optional
            Number of trials for sensitivity matrix (default: 2)

        Returns:
        --------
        list
            List of unique XandR arrays sorted by species count, then reaction count
        """
        cdef:
            int m, n, M = self.stoi.shape[0], N = self.stoi.shape[1]
            int idx
            cnp.ndarray[double, ndim=2] DT = np.array(self.A[N:, :M])
            bool_t[:] XandR = np.zeros(shape=(M + N), dtype=np.bool_)
            bool_t[:, :] lo_keeper = self._smat_(trial)
            list isoBSs = [], unique_isoBSs = []

        # Search starting from each reaction and optionally one species
        for m in range(-1, M):
            for n in range(N):
                XandR[:] = False
                if m >= 0:
                    XandR[m] = True
                XandR[M + n] = True
                XandR = cy_isoBS_searcher(XandR, self.__oc_keeper, lo_keeper, self.__cq_class, M, N)
                idx = cy_chiindex(self.stoi, DT, XandR)
                if idx == 0:
                    if cy_no_emergent_cycles(self.stoi, XandR):
                        isoBSs.append(np.array(XandR))

        # Default to full network if no isoBSs found
        if not isoBSs:
            XandR[:] = True
            isoBSs.append(np.array(XandR))

        # Filter unique subnetworks
        for g in isoBSs:
            flag_unique = True
            for movedg in unique_isoBSs:
                if np.all(g == movedg):
                    flag_unique = False
                    break
            if flag_unique:
                unique_isoBSs.append(g)

        # Sort by number of species, then reactions (ascending order)
        for m in range(len(unique_isoBSs) - 1):
            for n in range(m + 1, len(unique_isoBSs)):
                m_species = np.sum(unique_isoBSs[m][:M])
                n_species = np.sum(unique_isoBSs[n][:M])
                if m_species > n_species:
                    unique_isoBSs[m], unique_isoBSs[n] = unique_isoBSs[n], unique_isoBSs[m]
                elif m_species == n_species:
                    m_reacts = np.sum(unique_isoBSs[m][M:])
                    n_reacts = np.sum(unique_isoBSs[n][M:])
                    if m_reacts > n_reacts:
                        unique_isoBSs[m], unique_isoBSs[n] = unique_isoBSs[n], unique_isoBSs[m]

        return unique_isoBSs

    def signdet_subnetwork(self, XandR):
        """
        Compute the sign of the determinant for a subnetwork defined by XandR.

        Parameters:
        -----------
        XandR : array-like
            Boolean array (num_chem + num_react) indicating selected species and reactions

        Returns:
        --------
        float
            Sign of the determinant of the subnetwork A matrix
        """
        cdef:
            int M = self.stoi.shape[0]    # Number of chemical species
            int N = self.stoi.shape[1]    # Number of reactions
            double[:, :] Agamma           # Subnetwork A matrix

        # Count selected species and reactions
        M1 = np.sum(XandR[:M])
        N1 = np.sum(XandR[M:])

        # Construct subnetwork A matrix from selected species and reactions
        Agamma = cy_construct_A_from(
            self.stoi[XandR[:M], :][:, XandR[M:]],
            self.reginfo[XandR[:M], :][:, XandR[M:]]
        )
        # Compute and return sign of the determinant
        return signdet(np.array(Agamma))

    def structural_reduction(self, XandR):
        """
        Perform structural reduction on the CRN based on selected species and reactions.

        Parameters:
        -----------
        XandR : array-like
            Boolean array (num_chem + num_react) indicating species and reactions to keep

        Notes:
        ------
        Updates the CRN object in-place by reducing the stoichiometry matrix, regulation info,
        and derived properties based on the selected subnetwork.
        """
        cdef:
            int M = self.stoi.shape[0]    # Number of chemical species
            int N = self.stoi.shape[1]    # Number of reactions
            cnp.ndarray[double, ndim=2] NU = self.stoi
            cnp.ndarray[bool_t, ndim=1] X, R

        # Extract selected species and reactions
        X = np.array(XandR[:M], dtype=np.bool_)
        R = np.array(XandR[M:], dtype=np.bool_)

        # Compute reduced stoichiometry matrix
        # NU[~X, ~R] - NU[~X, R] * pinv(NU[X, R]) * NU[X, ~R]
        stoi_reduc = (
            NU[~X, :][:, ~R] -
            NU[~X, :][:, R].dot(np.linalg.pinv(NU[X, :][:, R])).dot(NU[X, :][:, ~R])
        )

        # Update CRN attributes
        self.stoi = stoi_reduc
        self.reginfo = self.reginfo[~X, :][:, ~R]          # Reduce reginfo to match kept species/reactions
        self.A = cy_construct_A_from(self.stoi, self.reginfo)
        self.__cq_class = np.array(self.__cq_class)[~X, :][:, ~X]    # Reduce CQ class for kept species
        # self.__cq_class = cy_cq_classifier(np.array(self.A[self.stoi.shape[1]:, :self.stoi.shape[0]]))
        self.__oc_keeper = np.array(self.__oc_keeper)[~X, :][:, ~R]  # Reduce oc_keeper accordingly
        self.X_names = [self.X_names[m] for m in range(M) if not XandR[m]]
        self.R_names = [self.R_names[n] for n in range(N) if not XandR[M + n]]

        return None

def indicator_diagnose(crn):
    assert isinstance(crn, CRN)
    res = []
    P = crn.stoi.shape[1]
    for p in range(P):
        M, N = crn.stoi.shape
        minisoBS = crn.search_for_isoBSs()[0]
        X = [crn.X_names[m] for m in range(len(crn.X_names)) if minisoBS[m]]
        R = [crn.R_names[n] for n in range(len(crn.R_names)) if minisoBS[M + n]]
        if len(X) == crn.stoi.shape[0]:
            minisoBS[:] = True
            X = [crn.X_names[m] for m in range(len(crn.X_names)) if minisoBS[m]]
            R = [crn.R_names[n] for n in range(len(crn.R_names)) if minisoBS[M + n]]
        sign_detA = crn.signdet_subnetwork(minisoBS)
        res.append( {'chemicals': X, 'reactions': R, 'sign_det': sign_detA} )
        if len(X) == crn.stoi.shape[0]:
            break
        crn.structural_reduction(minisoBS)
    return res

# ==== [ END OF THE SCRIPT ] ======================================================================================== #
