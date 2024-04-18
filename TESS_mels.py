# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 16:56:51 2023

@author: hyeseon
"""


import librosa

import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# TESS 데이터셋
TESS = "./dataset/TESS/"

tess_dir_list = os.listdir(TESS)
path_list = []
emotion_list = []

emotion_dic = {
    'happy' : 'happy',
    'neutral' : 'neutral',
    'sad' : 'sad',
    'angry' : 'angry',
    'fear' : 'fear',
    'disgust' : 'disgust',
    'ps' : 'pleasantSurprised'
    }

for directory in tess_dir_list:  
    audio_files = os.listdir(os.path.join(TESS, directory))
    for audio_file in audio_files:
        part = audio_file.split('.')[0]
        key = part.split('_')[2]
        if key in emotion_dic:
            path_list.append(f"{TESS}{directory}/{audio_file}")
            emotion_list.append(emotion_dic[key])
            
df = pd.concat([
    pd.DataFrame(path_list, columns=['path']),
    pd.DataFrame(emotion_list, columns=['emotion'])
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


mel_s = []

def GetMel_S():
    for i in path_list:
        mel_spec = Mel_S(i)
        mel_s.append([mel_spec])


#1차원 배열로 mel_s[n][0] n=len(mel_s)
#np.ravel(mel_s[0][0], order="F")

mel_spec = []
for i in path_list[:10]:
    mel_spec.append(Mel_S(i))


