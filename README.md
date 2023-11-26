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

TRIAL 1

For a small dataset with the keywords: 

['backward', 'bed', 'cat', 'follow']

['backward', 'bed', 'cat', 'follow', 'forward', 'learn']

['backward', 'bed', 'cat', 'follow', 'forward', 'learn', 'marvin', 'tree', 'visual', 'wow']

|Number of classes	|Number of clauses per class	|PyTM-MEL|PyTM-MFCC|TMU-MEL|TMU-MFCC|
|---      			|---							|---	 |---      |---    |---     |
|4					|500							|75.58	 |         |       |        |
|4					|1000							|75.72	 |         |       |        |
|4					|2000							|74.21	 |         |       |        |
|6					|1000							|62.86	 |         |       |        |
|6					|2000							|64.11	 |         |       |        |
|10 				|1000							|54.28   |         |       |        |
|10 				|2000							|54.67   |         |       |        |


TRIAL 2

For a bigger dataset with the keywords:

['yes' , 'no', 'stop', 'seven']

['yes' , 'no', 'stop', 'seven', 'left', 'right']

['yes' , 'no', 'stop', 'seven', 'left', 'right', 'up', 'down', 'backward', 'forward']

|Number of classes	|Number of clauses per class	|PyTM-MEL|PyTM-MFCC|TMU-MEL|TMU-MFCC|
|---      			|---							|---	 |---      |---    |---     |
|4					|500							|82.41	 |87.64    |81.76  |86.53   |
|4					|1000							|81.65 	 |88.33    |82.91  |86.98   |
|4					|2000							|81.53	 |88.62    |83.35  |87.42   |
|6					|1000							|70.61	 |81.20    |72.40  |80.29   |
|6					|2000							|70.05	 |82.42    |73.98  |81.29   |
|10 				|1000							|60.55   |71.29    |62.33  |70.39   |
|10 				|2000							|57.52   |71.57    |63.50  |71.47   |


We can see that we get better accuracy with a larger dataset.
