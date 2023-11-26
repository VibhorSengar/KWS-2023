import logging
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler

import numpy as np
from sklearn.model_selection import train_test_split
import pickle

_LOGGER = logging.getLogger(__name__)

def percentile_binning(matrix, percentile=50):
    threshold_value = np.percentile(matrix, percentile)
    result_matrix = np.zeros_like(matrix)
    result_matrix[matrix >= threshold_value] = 1
    return result_matrix

if __name__ == "__main__":
	nk = [4, 6, 10]
	for num_keywords in nk:
		# X = np.load(f"./complete_data_npy/subset_{nk}/X_train_{nk}.npy")
		# Y = np.load(f"./complete_data_npy/subset_{nk}/Y_train_{nk}.npy")
		# X = np.load(f"./dataset/data{num_keywords}/X_train_{num_keywords}.npy")
		# Y = np.load(f"./dataset/data{num_keywords}/y_train_{num_keywords}.npy")

		# X = percentile_binning(X)

		# X_flat = X.reshape(X.shape[0], -1) 
		# print(X.shape)
		# X_train, X_test, Y_train, Y_test = train_test_split(X_flat, Y, test_size=0.2, random_state=8)  


		X_train=np.load(".//dataset//split_data_npy//subset_"+str(num_keywords)+"//X_train.npy")
		X_test=np.load(".//dataset//split_data_npy//subset_"+str(num_keywords)+"//X_test.npy")
		Y_train=np.load(".//dataset//split_data_npy//subset_"+str(num_keywords)+"//Y_train.npy")
		Y_test=np.load(".//dataset//split_data_npy//subset_"+str(num_keywords)+"//Y_test.npy")

		X_train=percentile_binning(X_train)
		X_test=percentile_binning(X_test)

		print((X_train).shape)
		print((Y_train).shape)

		num_epochs = 50
		num_clauses = 1000
		T = 800
		s = 9.0
		max_included_literals = 32
		device = "CPU"
		weighted_clauses = True

		tm = TMClassifier(
		    type_iii_feedback=True,
		    number_of_clauses=num_clauses,
		    T=T,
		    s=s,
		    max_included_literals=max_included_literals,
		    platform=device,
		    weighted_clauses=weighted_clauses,
		    seed=42,
		)

		_LOGGER.info(f"Running {TMClassifier} for {num_epochs}")
		train_accuracies = []
		test_accuracies = []
		for epoch in range(num_epochs):
		    benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
		    with benchmark_total:
		        benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
		        with benchmark1:
		            res = tm.fit(
		                X_train,
		                Y_train,
		                metrics=["update_p"],
		            )
		        print(res)
		        benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
		        with benchmark2:
		            result = 100 * (tm.predict(X_test) == Y_test).mean()
		            r_train = 100 * (tm.predict(X_train) == Y_train).mean()

		        train_accuracies.append(r_train)
		        test_accuracies.append(result)

		        _LOGGER.info(f"Epoch: {epoch + 1}, , Train Accuracy: {r_train:.2f}, Test Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
		                    f"Testing Time: {benchmark2.elapsed():.2f}s")

		    if device == "CUDA":
		        CudaProfiler().print_timings(benchmark=benchmark_total)

		with open(f'./models/TMUmodel_{num_keywords}_{num_clauses}.pkl', 'wb') as file:
		    pickle.dump(tm, file)

		acc = np.array([train_accuracies, test_accuracies])
		with open(f'./accuracies/TMUmodel_{num_keywords}_{num_clauses}.pkl', 'wb') as file:
		    pickle.dump(acc, file)
		# (15855, 13, 21)
		# (7288, 28, 28)