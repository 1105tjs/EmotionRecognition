# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:12:35 2023

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

path_list = []
audio_dir = "./dataset/InsideOut/Emo/" # 폴더 경로
dir_list = os.listdir(audio_dir) # 폴더 안에 파일들


# Mel-Spectrogram

frame_length = 0.025
frame_stride = 0.010

def Mel_S(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)

    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram example.png')
    plt.show()

    return S

df = pd.read_csv('insideOut_Path_Emo_label.csv')

for path in df['path']:
    path_list.append(path)
    
mel_s = []

def GetMel_S():
    for i in path_list:
        mel_spec = Mel_S(i)
        mel_s.append([mel_spec])
        
m=[]

def PushDf():    
     for i in range(len(mel_s)):
        m.append(np.ravel(mel_s[i][0], order="F"))

            
mm = pd.DataFrame(m)   
df2 = pd.concat([df, mm], axis=1)

#df3.to_csv('insideOut_EmoChar_mel.csv', index='False')