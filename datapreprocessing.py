import os
import pandas as pd
import numpy as np
import librosa
from glob import glob

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    # Spectral constrat
    spect_contr = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spect_contr))

    return np.array(result)

path = './Dataset/ESD/'

files = glob(os.path.join(path, "**","**", "*.wav"), recursive=True)
print("Some files: ",files[0:4])

english = pd.DataFrame()
mandarin = pd.DataFrame()

english_labels = []
mandarin_labels = []
#import random
for i in range(len(files)):
    temp = pd.DataFrame(columns=['features'])
    data, sample_rate = librosa.load(files[i], duration=2.5, offset=0.5)
    label = files[i].split(path)[1].split('/')[1]
    speaker = int(files[i].split(path)[1].split('/')[0])
     
    features = extract_features(data=data, sample_rate=sample_rate)
    if speaker > 10:
        english = pd.concat((english, pd.DataFrame(features).T), axis=0, ignore_index=True)
        #english['label'] = label
        english_labels.append(label)
    else:
        mandarin = pd.concat((mandarin, pd.DataFrame(features).T), axis=0, ignore_index=True)
        #mandarin['label'] = label
        mandarin_labels.append(label)

    if i%1000 == 0:
        print(f'{i}/{len(files)}')

mandarin['label'] = mandarin_labels
english['label'] = english_labels

mandarin.to_csv('mandarin.csv', index=False)
english.to_csv('english.csv', index=False)
print('done')
