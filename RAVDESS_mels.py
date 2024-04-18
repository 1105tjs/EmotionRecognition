# -*- coding: utf-8 -*-

import librosa
import librosa.display
import pandas as pd
import numpy as np
import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 데이터셋 경로
dataset = "./dataset/RAVDESS/"
dir_list = os.listdir(dataset)
path_list = []
emotion_list = {
    '01' : 'neutral',
    '02' : 'calm',
    '03' : 'happy',
    '04' : 'sad',
    '05' : 'angry',
    '06' : 'fear',
    '07' : 'disgust',
    '08' : 'surprise'
    }
emotion = []

for directory in dir_list:  
    wav_files = os.listdir(os.path.join(dataset, directory))
    for wav_file in wav_files:
        part = wav_file.split('.')[0]
        key = part.split('-')[2]
        path_list.append(f"{dataset}{directory}/{wav_file}")
        if key in emotion_list:
            emotion.append(emotion_list[key])
        
  
df = pd.concat([
    pd.DataFrame(path_list, columns=['path']),
    pd.DataFrame(emotion, columns=['emotion'])
    ], axis=1)

# Mel-Spectrogram
frame_length = 0.025
frame_stride = 0.010

def Mel_S(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)

    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

    return S

mel_spec = []

for i in path_list:
    mel_spec.append(Mel_S(i))
    
mels_1d =[]

for i in range(len(mel_spec)):
    mels_1d.append(np.ravel(mel_spec[i][0], order="F"))
            
mm = pd.DataFrame(mels_1d)   
df2 = pd.concat([df, mm], axis=1)

#df2.to_csv('./dataset/RAVDESS_mels.csv', index='False')
