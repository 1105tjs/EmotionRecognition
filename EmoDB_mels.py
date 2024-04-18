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
dataset = "./dataset/EmoDB/"
file_list = os.listdir(dataset)
path_list = []


for wav_file in file_list:  
    wav_files = os.path.join(dataset, wav_file)
    path_list.append(f"{dataset}{wav_file}")
            
#파일명에서 감정 추출    
emo = []
for i in range(len(file_list)):
    emo.append(file_list[i][-6])
    
df = pd.concat([
    pd.DataFrame(path_list, columns=['path']),
    pd.DataFrame(emo, columns=['emotion'])
    ], axis=1)

df['emotion'] = df['emotion'].map({
    'W' : 'anger',
    'L' : 'boredom',
    'E' : 'disgust',
    'A' : 'fear',
    'F' : 'happiness',
    'T' : 'sadness',
    'N' : 'neutral'
    })

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

#df2.to_csv('./dataset/EmoDB_mels.csv', index='False')
