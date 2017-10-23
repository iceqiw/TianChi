import pandas as pd

userMallSet_m=pd.read_csv('data/userMallSet_m_615_origin.csv')

def extract_wifi_info(row):
    sigleList=row[-1].split(';')
    names=[]
    intensity=[]
    isConnect=[]
    for s in sigleList:
        oo=s.split('|')
        names.append(oo[0])
        intensity.append(oo[1])
        isConnect.append(oo[2])
    row[-1]=[names,intensity,isConnect]
    return row

te=userMallSet_m.apply(extract_wifi_info,axis=1)
# print(te.shape)
print(te.head()['wifi_infos'][1][1])