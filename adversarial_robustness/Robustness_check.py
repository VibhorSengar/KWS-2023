#from PyTsetlinMachineCUDA.tm import TsetlinMachine, MultiClassTsetlinMachine
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import numpy as np 
import sympy
from sympy import sympify
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from sympy.logic.boolalg import to_dnf, to_cnf
from sympy.logic.inference import satisfiable
import os
import timeit
import cv2
import pickle 
import sys
import time
sys.path.insert(1, '../')
from robustness_utils_new import *
import random

def percentile_binning(matrix, percentile=50):
    threshold_value = np.percentile(matrix, percentile)
    result_matrix = np.zeros_like(matrix)
    result_matrix[matrix >= threshold_value] = 1
    return result_matrix

number_of_classes = 10
n_clauses = 1000

f_tm = open("../trial1/best_tms/tm_"+str(number_of_classes)+"_"+str(n_clauses)+".obj", "rb")
tm = pickle.load(f_tm)
f_tm.close() 

X_test = np.load("../trial1/dataset/split_data_npy/subset_"+str(number_of_classes)+"/X_test.npy")

X_test = X_test.reshape(X_test.shape[0], -1) 
X_test=percentile_binning(X_test)


X_test= np.array(random.choices(X_test, k=100))


print(X_test.shape)

number_of_features = int(tm.number_of_features/2)
print(number_of_classes,n_clauses,number_of_features)

# Encoding tsetlin machine for each class.

counter = VarCounter()
x = [None] + define_variables(number_of_features, counter)
o = define_variables(1, counter)
posnegs = []
ts_encodings = []
for t in range(number_of_classes):
    pos, neg = get_clauses(tm,t, x)
    posnegs.append((pos,neg))
    n_clauses = len(pos) + len(neg) -2
    print("n_clause", n_clauses)
    encoded = encode_test(neg, pos, n_clauses,counter, o)
    ts_encodings.append(encoded)



predict= tm.predict(X_test)
tot=0


        
print("Encoding Finished")

# Verifying robustness for epsilon = {1, 3, 5} and saving the results in statistics_robustness.

ps = [1, 3, 5]
statistics_robustness = np.zeros((len(ps),4))
from time import time
for pix,p in enumerate(ps):
    print("ps", pix, p)
    for i in range(100):
        print("test", i)
        runtimes, runtimef,rob = check_robustness(p, x, ts_encodings[predict[i]], X_test[i], 1,counter, o)
        print ("class",predict[i],"rob_solve_time", runtimes)
        statistics_robustness[pix][0]+=runtimes
        statistics_robustness[pix][1]+=runtimef
        statistics_robustness[pix][2]+=rob
        if runtimes<=300:
            statistics_robustness[pix][3]+=1

    statistics_robustness[pix][0]=statistics_robustness[pix][0]/100
    statistics_robustness[pix][1]=statistics_robustness[pix][1]/100
    print(statistics_robustness[pix])

print("saving")        
f = open("./robustness_result_"+str(number_of_classes)+"_"+str(n_clauses)+".pickle", "wb+")
pickle.dump(statistics_robustness, f)
f.close()

