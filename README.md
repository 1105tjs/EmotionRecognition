# 딥러닝 모델을 이용한 음성 감정 인식
음성 감정 데이터 세트로부터 감정을 학습하고 영화 등장인물의 감정을 인식하여 감성 자막을 표현합니다.


## 개발 환경
- TensorFlow 2.9.2
- Python 3.10.11
- Intel Core i9 3.30GHz processor
- 128GB of RAM
- two NIVIDIA GeForce RTX 3090

## 데이터셋
  - 훈련 데이터로 Kaggle에 제공된 4가지 음성 감정 데이터셋을 활용했습니다
  - https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?rvi=1
  - https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb?rvi=1
  - https://www.kaggle.com/datasets/barelydedicated/savee-database
  - https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

## 파일 설명
##### 1. Mel-Spectrogram 변환 - 데이터셋의 음성 파일(.wav)을 멜 스펙트로그램으로 변환한 후 csv로 저장합니다
	TESS_mels.py
	SAVEE_mels.py
	RAVDESS_mels.py
	EmoDB_mels.py
	inside_mels.py
	inside_label.py - insideOut의 음성 파일명에서 캐릭터 이름을 추출해 캐릭터명을 감정으로 라벨링 하여 csv 파일로 저장

##### 2. 데이터 학습 - 데이터셋을 딥러닝 모델에 학습합니다
	TESS.py
	SAVEE.py
	RAVDESS.py
	EmoDB.py

##### 3. 전이학습 - 4가지 음성 데이터로 훈련된 각각의 모델에 영화 insideOut 음성 데이터를 학습시킵니다.
	TESS_inside.py
	SAVEE_inside.py
	RAVDESS_inside.py
	EmoDB_inside.py

##### 4. 결과 확인 - confusion matrix, F1-Score 확인하고 저장합니다.
	TESS_inside_cm.py
	SAVEE_inside_cm.py
	RAVDESS_inside_cm.py
	EmoDB_inside_cm.py
