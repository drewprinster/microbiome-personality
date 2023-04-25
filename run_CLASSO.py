import numpy as np
import pandas as pd
import scipy
import matplotlib
from CLASSO_modified.classo import *
import time
import sys
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    filter_level = sys.argv[1]
    threshold = float(sys.argv[2])
    
    print("CLASSO_" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3])

    startTime = time.time()

    # FULL MODEL for CSHL V2 with labels, ** selection frequency threshhold = 0.7 **
    # HEXACO ~ ASVs_(filter_level% filter) + Sex + Age + SNPs_personality + SNPs_microbiome
    
    Y_df = pd.read_csv("20220325_CLASSO_Y_" + filter_level + ".csv", index_col = 0)
    Y_full = Y_df.to_numpy()
    
    X_df = pd.read_csv("20220325_CLASSO_X_" + filter_level + ".csv", index_col = 0)
    X_full = X_df.to_numpy()
    
    C = np.load("20220325_CLASSO_C_" + filter_level + ".npy")
#     num_ASVs = int(np.sum(C))


    for i in range(0, 30):
        non_nan = ~np.isnan(Y_full[:, i])
        y = Y_full[:, i][non_nan]
        y = y - np.mean(y)
        X = X_full[non_nan]
        n, p = np.shape(X)
#         scaler = StandardScaler()
#         X = scaler.fit_transform(X) ## Scale 

        # print variable name
        print("   ")
        print(Y_df.columns[i])
        print("X shape = ", np.shape(X))
        print("Y shape = ", np.shape(y))
        print("C shape ", np.shape(C))
        print("   ")


        # *** create classo problem for 0.01 percent median
        problem = classo_problem(X, y, C, label=X_df.columns)
        problem.data.X = X
        problem.data.y = y
        problem.data.C = C
        problem.data.label = X_df.columns

        ## ADDED
        problem.numerical_method = 'Path-Alg'
        problem.formulation.concomitant = True
        problem.formulation.huber = False

        # Solve the stability selection : (by default, it will use the theoritical lambda)
        problem.model_selection.StabSel                       = True
        problem.model_selection.StabSelparameters.method      = 'lam'

        # Added by DREW
        problem.model_selection.StabSelparameters.numerical_method = 'Path-Alg'
        problem.model_selection.StabSelparameters.lamin      = 0.01
        problem.model_selection.StabSelparameters.threshold      = threshold
        problem.model_selection.StabSelparameters.B     = 200
        problem.model_selection.StabSelparameters.q     = round(np.sqrt((2*threshold-1)*p))


        print(problem)

        problem.solve()
        
        t_args = ["0.6", "0.7", "0.8", "0.9"]
        t_save = ["06", "07", "08", "09"]
        t = "error"
        for j in range(0, 4):
            if sys.argv[2] == t_args[j]:
                t = t_save[j]

        # problem.solution.PATH.save = 'R3-'
        problem.solution.StabSel.save1 = '/home/ajp65/project/TwinsUK/CLASSO_' + sys.argv[1] + '_t' + t + '_q' + sys.argv[3] + '/' + str(Y_df.columns[i]) + '_R3-StabSel'
        problem.solution.StabSel.save2 = '/home/ajp65/project/TwinsUK/CLASSO_' + sys.argv[1] + '_t' + t + '_q' + sys.argv[3] + '/' + str(Y_df.columns[i]) + '_R3-StabSel-beta'

        print(problem.solution)

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
