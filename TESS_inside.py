import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #gpu 사용
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau, ModelCheckpoint

df = pd.read_csv('insideOut_EmoChar_mel_onlyRmv.csv')
df.drop(['Unnamed: 0','Unnamed: 0.1','path'], axis=1, inplace=True)
df['label'] = df['label'].map({
    'JOY' : 0,
    'SADNESS' : 1,
    'ANGER' :2,
    'FEAR' : 3,
    'DISGUST' : 4})
df = df.fillna(0)

input_data = df.drop(['label'],axis=1)
target_data = df['label'].to_numpy()

time_steps= input_data.shape[1] // 40
input_data_array = input_data.values
input_data_3d = np.reshape(input_data_array, (input_data_array.shape[0], time_steps, 40))

train_input, test_input, train_target, test_target = train_test_split(input_data_3d, target_data, train_size=0.7, random_state=42)
test_input, val_input, test_target, val_target = train_test_split(test_input,test_target,test_size=0.5,random_state=42)

ss = StandardScaler()
train_input = ss.fit_transform(train_input.reshape(-1, train_input.shape[-1])).reshape(train_input.shape)
test_input = ss.fit_transform(test_input.reshape(-1, test_input.shape[-1])).reshape(test_input.shape)
val_input = ss.transform(val_input.reshape(-1, val_input.shape[-1])).reshape(val_input.shape)

#콜백함수
filename = 'checkpoint-{epoch:02d}.h5'
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=30)
callbacks_list = [reduceLR, checkpoint, early_stop]

model = load_model('./TESS_model')

model.fit(train_input, train_target, epochs=200, batch_size=32,validation_data=(val_input, val_target))

model.save('./TESS_inside')

