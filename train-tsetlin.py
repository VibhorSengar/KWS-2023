# from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import random

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# from keras.datasets import mnist

def percentile_binning(matrix, percentile=50):
    threshold_value = np.percentile(matrix, percentile)
    result_matrix = np.zeros_like(matrix)
    result_matrix[matrix >= threshold_value] = 1
    return result_matrix

number_of_classes=4
number_of_clauses=1000
X = np.load(".//dataset//X_train_"+str(number_of_classes)+".npy")
Y = np.load(".//dataset//Y_train_"+str(number_of_classes)+".npy")
# num_samples = X.shape[0]
# indices = np.random.permutation(num_samples)
# X = X[indices]
# Y = Y[indices]
# smaller=num_samples//4
# X = X[:smaller]
# Y = Y[:smaller]
# print(X.shape)
# print(Y.shape)

#visualise data
# keyword_dir=['backward', 'bed', 'cat', 'follow', 'forward', 'learn', 'marvin', 'tree', 'visual', 'wow']
# comp=[]
# for idx in range(10):
# 	indices=np.where(Y==idx)[0][:5]
# 	temp=X[indices]
# 	comp.append(temp)

# # create figure 
# fig = plt.figure(figsize=(10,10)) 
  
# setting values to rows and column variables 
# rows = 10
# columns = 6

# idx=1
# i=0
# for keyword in comp:
# 	fig.add_subplot(rows, columns, idx) 
# 	plt.axis('off') 
# 	plt.title(keyword_dir[i]) 
# 	idx+=1
# 	for spect in keyword:
# 		# Adds a subplot at the 1st position 
# 		fig.add_subplot(rows, columns, idx) 
		  
# 		# showing image 
# 		plt.imshow(spect) 
# 		plt.axis('off') 
# 		# plt.title(keyword_dir[i]) 

# 		idx+=1
# 	i+=1
# plt.show()

#get the datasets ready
X_flat = X.reshape(X.shape[0], -1) 

X_train, X_test, Y_train, Y_test = train_test_split(X_flat, Y, test_size=0.1, random_state=8)  

# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# X_train = np.where(X_train > 75, 1, 0) 
# X_test = np.where(X_test > 75, 1, 0) 

X_train=percentile_binning(X_train)
X_test=percentile_binning(X_test)

print((X_train).shape)
print((Y_train).shape)

# tm = MultiClassTsetlinMachine(1000, 50, 10.0)
# tm = MultiClassTsetlinMachine(number_of_clauses=1000, T=17, s=5.0)
tm = MultiClassTsetlinMachine(number_of_clauses=number_of_clauses, T=17, s=5.0, clause_drop_p = 0.05, literal_drop_p=0.0)

# print("\nAccuracy over 10 epochs:\n")
# for i in range(10):
# 	start_training = time()
# 	tm.fit(X_train, Y_train, epochs=1, incremental=True)
# 	stop_training = time()

# 	# print('train done')
# 	start_testing = time()
# 	result = 100*(tm.predict(X_test) == Y_test).mean()
# 	stop_testing = time()

# 	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
num_samples = X_train.shape[0]
print("\nAccuracy over 50 epochs:\n")
for i in range(50):
	start_training = time()
	indices = np.random.permutation(num_samples)
	X_train = X_train[indices]
	Y_train = Y_train[indices]

	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))