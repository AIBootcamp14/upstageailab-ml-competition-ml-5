#%%
# python 3.10
# see requirements.txt for package versions

# import package needed
# %pip install -r requirements.txt

#%%
# import package needed
import os
import tarfile
import wget
import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# set kr font
import matplotlib.font_manager
fontlist = [font.name for font in matplotlib.font_manager.fontManager.ttflist]
# print(fontlist)
krfont = ['Malgun Gothic', 'NanumGothic', 'AppleGothic']
for font in krfont:
    if font in fontlist:
        mpl.rc('font', family=font)

# make data folder
rawdata_dir = 'rawdata'
os.makedirs(rawdata_dir, exist_ok=True)


#%% download and unzip original data
basefile_url = 'https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000363/data/data.tar'
basefile_filename = 'basefile.tar'
basefile_path = os.path.join(rawdata_dir, basefile_filename)

if not os.path.exists(basefile_path):
    wget.download(basefile_url, basefile_path)
    with tarfile.open(basefile_path, 'r') as tar:
        tar.extractall(path=rawdata_dir)
else: pass


#%%
# import & save external data - 대출금리 데이터

loanrate_filename = 'loanrate.csv'
loanrate_path = os.path.join(rawdata_dir, loanrate_filename)

if not os.path.exists(loanrate_path):
    # url을 통해 json 데이터 가져오기
    # KOSIS API-key가 필요합니다.
    url = 'API-key값이 들어간 URL'
    with urlopen(url) as url:
        json_file = url.read()
        
    py_json = json.loads(json_file.decode('utf-8'))

    month_list = []
    loanrate_list = []

    for i in py_json:
        if i['C1_NM'] == '가계대출':
            month_list.append(i['PRD_DE'])
            loanrate_list.append(i['DT'])

    loanrate_df = pd.DataFrame({'month': month_list, 'loanrate': loanrate_list})
    loanrate_df.to_csv(loanrate_path, index=False)

        
# display(loanrate_df)
# display(py_json[:5])


#%%
# import & save external data - 인구수 데이터
# 년단위 자료입니다 (월별자료는 2011년부터 존재)

population_filename = 'population.csv'
population_path = os.path.join(rawdata_dir, population_filename)

if not os.path.exists(population_path):
    # url을 통해 json 데이터 가져오기
    # KOSIS API-key가 필요합니다.
    url = 'API-key값이 들어간 URL'
    with urlopen(url) as url:
        json_file = url.read()
        
    py_json = json.loads(json_file.decode('utf-8'))

    year_list = []
    area_list = []
    pop_list = []
    class_list = []

    for i in py_json:
        if i['C1_NM'] != '서울특별시':
            year_list.append(i['PRD_DE'])
            area_list.append(i['C1_NM'])
            pop_list.append(i['DT'])
            class_list.append(i['ITM_NM'])

    population_df = pd.DataFrame({'year': year_list, 'area': area_list, 'class': class_list, 'population': pop_list})
    population_df.to_csv(population_path, index=False)

# display(loanrate_df)
# display(py_json[100:102])


#%%
# import raw datasets
train_rawdf = pd.read_csv('./rawdata/train.csv', dtype = {16: 'str', 17: 'str', 36: 'str'})
test_rawdf = pd.read_csv('./rawdata/test.csv')
bus_rawdf = pd.read_csv('./rawdata/bus_feature.csv')
subway_rawdf = pd.read_csv('./rawdata/subway_feature.csv')

loanrate_df = pd.read_csv('./rawdata/loanrate.csv')
population_df = pd.read_csv('./rawdata/population.csv')


print(train_rawdf.head(), "\n")
print(train_rawdf.columns, "\n")
print(test_rawdf.head(), "\n")
print(bus_rawdf.head(), "\n")
print(subway_rawdf.head(), "\n")

print(loanrate_df.head(), "\n")
print(population_df.head(), "\n")






#%%
# export package used
# %pip freeze > requirements.txt