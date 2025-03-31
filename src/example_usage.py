
import crnindsp
import numpy as np
import pandas as pd

# Example 1: a singular network.
print("Given the stoichiometric matrix (nu)")
nu = np.array([[ -1,  0,  0,  1,  0],
               [  0, -1,  0,  0,  1],
               [ -1, -1,  2,  0,  0],
               [  1,  0, -1,  0,  0],
               [  0,  1, -1,  0,  0],], dtype = float)
print(nu)
print("without specifying the regulatory information,")
print("one can use crnindsp.CRN(nu) to construct a chemical reaction network object,")
print("and the A-matrix is given by crnindsp.CRN(nu).A as")
crn = crnindsp.CRN(nu)
A   = crn.A
print(A)
print("By putting the CRN class object as the input of crnindsp.indicator_diagnose(),")
print("one can obtains a structural decomposition")
decomp = crnindsp.indicator_diagnose(crn)
DF = pd.DataFrame(decomp)
DF.index = DF.index[::-1] # this step is not necessary
print(DF)

# Example 2: a regular network
print("Similarly, given a stoichiometric matrix")
S = np.array([[ -1,  1,  0,  0,  0,  0,  0,  0,],
              [  1, -1, -1,  1, -1,  0,  0,  0,],
              [  0,  0,  1, -1, -1,  3,  0,  0,],
              [  0,  0,  0,  0,  0, -3,  0,  1,],
              [  0,  0,  0,  0,  1,  0, -1,  0,],], dtype = float)
print(S)
print("and this time we further consider the regulatory information given by")
reginfo = S.copy() * 0
reginfo[S < 0] = np.inf # since reactants should regulate the reactions positively
# and we put
reginfo[0, 1]  = np.nan # when assuming that chemical X0 can regulate reaction R1 positively and negatively
reginfo[4, 7]  = np.inf # when assuming that chemical X4 regulates reaction R7 positively
# Of course, one can assign (-inf) to an entry for a negative regulation.
print(reginfo)

# Let's name the chemicals
chem_names = ['A', 'B', 'C', 'D', 'E']

# Then, the chemical reaction network is constructed by putting
crn5sp = crnindsp.CRN(S, reginfo, X_names = chem_names) # and R_names if you want to name the reactions.
# The A-matrix is
A_5sp  = crn5sp.A
print("Then the A-matrix is")
print(A_5sp.round(2))

# And we can analoguously do the analysis:
decomp = crnindsp.indicator_diagnose(crn5sp)
DF5sp = pd.DataFrame(decomp)
DF5sp.index = DF5sp.index[::-1]
print("The structural decomposition is")
print(DF5sp)
# idx_eliminable = (np.sign(DF5sp.sign_det).abs() - 1).abs() < 1e-9
# print(DF5sp.loc[~idx_eliminable, :])

