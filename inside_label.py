# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:58:48 2023

@author: hyeseon
"""

import os
import pandas as pd

# 폴더 내 파일 리스트를 얻기 위한 경로 설정
folder_path = './dataset/InsideOut/RemoveVer'

# 파일 리스트 얻기
file_list = os.listdir(folder_path)

# 추출된 정보를 저장할 리스트
file_paths = []
extracted_strings = []

# 파일 리스트를 순회하며 파일 경로와 확장자 앞의 문자열 추출하여 저장
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)  # 파일 경로
    
    # 파일 이름에서 확장자 분리
    file_name_without_extension, file_extension = os.path.splitext(file_name)
    under = '_'
    vocals=' [vocals]'
    # 확장자가 있는 경우에만 추출하여 저장
    if file_extension:
        extracted_string = file_name_without_extension.split('.')[-1]  # 확장자 앞의 문자열 추출
        
        # 추출된 문자열 앞뒤에 있는 공백 제거
        extracted_string = extracted_string.strip()
    if under:
        extracted_string = extracted_string.split('_')[0]
    if vocals:
        extracted_string = extracted_string.split(' [vocals]')[0]
    else:
        extracted_string = None
    
    file_paths.append(file_path)
    extracted_strings.append(extracted_string)

# 추출된 정보들을 데이터프레임에 저장
df = pd.DataFrame({'path': file_paths, 'label': extracted_strings})
df.to_csv('insideOut_Path_All_label.csv', index='False')
# 결과 확인
print(df)

df2 =  pd.read_csv('./insideOut_Path_All_label.csv', encoding='ISO-8859-1')

df2 = df2[df2['label'].isin(['SADNESS', 'JOY', 'FEAR', 'DISGUST', 'ANGER'])]
df2.to_csv('insideOut_Path_Emo_label.csv', index='False')