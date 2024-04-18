from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns


df = pd.read_csv('insideOut_EmoChar_mel_onlyRmv.csv')
df.drop(['Unnamed: 0','Unnamed: 0.1', 'path'], axis=1, inplace=True)
df['label'] = df['label'].map({
    'JOY' : 1,
    'SADNESS' : 4,
    'ANGER' :3,
    'FEAR' : 5,
    'DISGUST' : 6})
df = df.fillna(0)

input_data = df.drop(['label'],axis=1)
target_data = df['label'].to_numpy()

time_steps= input_data.shape[1] // 40
input_array = input_data.values
input_3d = np.reshape(input_array, (input_array.shape[0], time_steps, 40))

train_input, test_input, train_target, test_target = train_test_split(input_3d, target_data, train_size=0.7, random_state=42)
test_input, val_input, test_target, val_target = train_test_split(test_input,test_target,test_size=0.5,random_state=42)

ss = StandardScaler()
train_input = ss.fit_transform(train_input.reshape(-1, train_input.shape[-1])).reshape(train_input.shape)
test_input = ss.fit_transform(test_input.reshape(-1, test_input.shape[-1])).reshape(test_input.shape)
val_input = ss.transform(val_input.reshape(-1, val_input.shape[-1])).reshape(val_input.shape)
input_scaled = ss.transform(input_3d.reshape(-1, input_3d.shape[-1])).reshape(input_3d.shape)

model = load_model('./RAVDESS_insideRmv_model')

predicted_labels = model.predict(input_scaled)
predicted_labels = np.argmax(predicted_labels, axis=1)

class_names = ['JOY', 'SADNESS', 'FEAR', 'DISGUST', 'ANGER']


# 감정 캐릭터별 발화 비율 계산
total_samples = len(df)
emotion_ratios = df['label'].value_counts(normalize=True).sort_index()
emotion_ratios.index = class_names
emotion_ratios.name = "감정 캐릭터별 발화 비율"
print(emotion_ratios)
print("")

# 모델을 통한 감정 예측 비율 계산
predicted_ratios = pd.Series(predicted_labels).value_counts(normalize=True).sort_index()
predicted_ratios.index = class_names
predicted_ratios.name = "모델을 통한 감정 예측 비율"
print(predicted_ratios)

# inside-out 데이터의 캐릭터별 발화 수에 대한 Confusion Matrix 생성
# 같은 감정 클래스 간에 예측 결과와 실제 값이 동일하므로 모든 예측은 올바르게 됨
conf_matrix1 = confusion_matrix(df['label'], df['label'])
print("\ninside-out 데이터의 캐릭터별 발화 수에 대한 Confusion Matrix:")
print(conf_matrix1)

# inside-out 데이터의 캐릭터 발화 비율과 모델 예측 비율에 대한 Confusion Matrix 시각화
conf_matrix2 = confusion_matrix(target_data, predicted_labels)
print("\ninside-out 데이터의 캐릭터 발화 비율과 모델 예측 비율에 대한 Confusion Matrix:")
print(conf_matrix2)

# 분류 보고서 출력
class_names = ['JOY', 'SADNESS', 'FEAR', 'DISGUST', 'ANGER']
print("\nRAVDESS-insideOut(rmv) Classification Report:")
print(classification_report(target_data, predicted_labels, target_names=class_names))

# 정확도 : 전체 샘플 중 올바르게 분류된 샘플의 비율
accuracy = accuracy_score(target_data, predicted_labels)
print("Accuracy:", accuracy)

# 정밀도 : 모델이 양성으로 예측한 샘플 중에서 실제로 양성인 샘플의 비율
precision = precision_score(target_data, predicted_labels, average='weighted')
print("Precision:", precision)

# 재현율 : 실제 양성인 샘플 중에서 모델이 양성으로 예측한 샘플의 비율
recall = recall_score(target_data, predicted_labels, average='weighted')
print("Recall:", recall)

# f1 점수 : 정밀도와 재현율의 조화 평균으로 계산되는 값
f1 = f1_score(target_data, predicted_labels, average='weighted')
print("F1 Score:", f1)

# inside-out 데이터의 캐릭터별 발화 수에 대한 Confusion Matrix 시각화
# plot_confusion_matrix(conf_matrix, target_names=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix1, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('RAVDESS-insideOut Confusion Matrix')
plt.savefig('./RAVDESS_insideRmv_cm1.png')

# inside-out 데이터의 캐릭터 발화 비율과 모델 예측 비율에 대한 Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('RAVDESS-insideOut Confusion Matrix')
plt.savefig('./RAVDESS_insideRmv_cm2.png')

# Normalized Confusion Matrix 시각화
norm_conf_matrix = conf_matrix2.astype('float') / conf_matrix2.sum(axis=1)[:, np.newaxis]
norm_conf_matrix = np.clip(norm_conf_matrix, 0, 1)  # 값의 범위를 [0, 1]로 제한
plt.figure(figsize=(8, 6))
sns.heatmap(norm_conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("RAVDESS-insideOut Normalized Confusion Matrix")
plt.savefig('./RAVDESS_insideRmv_ncm.png')
