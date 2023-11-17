import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize 
from sklearn.model_selection import train_test_split
import os

# keyword_dir=['backward', 'bed', 'cat', 'follow', 'forward', 'learn', 'marvin', 'tree', 'visual', 'wow']
path='.//speech_command'
dir_list = os.listdir(path)
X=[]
Y=[]
dir_list.sort()
#should be less than 10
num_classes=1
for keyword_idx in range(num_classes):
	keyword=dir_list[keyword_idx]
	keyword_path=path+'//'+keyword
	audio_file_list=os.listdir(keyword_path)
	print(keyword, " ", keyword_idx)

	for audio_file in audio_file_list:
		# Load the .wav file
		# audio_file = '0a2b400e_nohash_0.wav'  # Replace with the path to your .wav file
		# print(audio_file)
		audio_file=path+'//'+keyword+'//'+audio_file
		y, sr = librosa.load(audio_file)

		# Calculate the Mel spectrogram
		sgram = librosa.stft(y)
		sgram_mag, _ = librosa.magphase(sgram)
		mel_spectrogram = librosa.feature.melspectrogram(S=sgram_mag, fmax=8000, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

		# Convert to decibels
		mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

		# Resize the Mel spectrogram to 28x28
		mel_spectrogram_resized = resize(mel_spectrogram_db, (28, 28))

		X.append(mel_spectrogram_resized)
		Y.append(keyword_idx)

X=np.array(X)
X = X.reshape(X.shape[0], -1)
Y=np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=8)

x_train_file_path=os.path.join('split_data_npy', 'subset_'+str(num_classes), 'X_train')
x_test_file_path=os.path.join('split_data_npy', 'subset_'+str(num_classes), 'X_test')
y_train_file_path=os.path.join('split_data_npy', 'subset_'+str(num_classes), 'Y_train')
y_test_file_path=os.path.join('split_data_npy', 'subset_'+str(num_classes), 'Y_test')

np.save(x_train_file_path, X_train)
np.save(x_test_file_path, X_test)
np.save(y_train_file_path, Y_train)
np.save(y_test_file_path, Y_test)


# # Load the .wav file
# audio_file = '0a2b400e_nohash_0.wav'  # Replace with the path to your .wav file
# y, sr = librosa.load(audio_file)

# # Calculate the Mel spectrogram
# sgram = librosa.stft(y)
# sgram_mag, _ = librosa.magphase(sgram)
# mel_spectrogram = librosa.feature.melspectrogram(S=sgram_mag, fmax=8000, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

# # Convert to decibels
# mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# # Resize the Mel spectrogram to 28x28
# mel_spectrogram_resized = resize(mel_spectrogram_db, (28, 28))

# # Display the Mel spectrogram
# print(mel_spectrogram_resized.shape)
# plt.figure(figsize=(6, 6))
# plt.imshow(mel_spectrogram_resized, cmap='grey', origin='lower')
# # plt.title('Mel Spectrogram')
# plt.axis('off')
# plt.show()

# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mel_spectrogram_resized, x_axis='time', y_axis='mel')
# plt.title('Mel Spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()
