# kws-2023

Here we train the Tsetline Machine Model given at https://github.com/cair/pyTsetlinMachine for Keyword Spotting.

# How to Use
1. Install the library at https://github.com/cair/pyTsetlinMachine along with the following:
	- NumPy
	- MatplotLib
	- ScikitLearn
	- ScikitImage
	- Librosa

2. To prepare the data set with a specific number of classes run the file `make_dataset_melspectro.py` in the `dataset` folder and specify the number of classes needed in the script.

3. To perform the training run the file `train-tsetlin.py` where we can specify the `number_of_classes` needed to train and the `number_of_clauses` per class parameters.

# Dataset

| Parameter                   | Value                |
|---                          | ---                  |
| Dataset                     | Speech-Command Subset|
| Feature Extraction          | MEL Spectrogram      |
| Library used                | Librosa              |
| Number of Frequency Bins    | 8000                 |
| FFT Window Length           | 2048                 |
| Hop Length                  | 512                  |


# Training

For pre-processing, the spectrograms are resized to 28x28 shape and a 50-percentile threshold binarization is done. We run a total of 50 epochs and following are the accuracies for the different values of the parameters

|Number of classes	|Number of clauses per class	|Accuracy|
|---      			|---							|---	 |
|4					|1000							|74.35	 |
|4					|2000							|75.86	 |
|6					|2000							|56.43	 |
|6					|4000							|58.45   |
|10 				|1000							|49.61	 |
|10 				|2000							|51.67	 |
|10    				|4000							|41.39   |


