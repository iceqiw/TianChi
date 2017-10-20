import pandas as pd
from datetime import *

def splitDate(mallid):
    userMallSet=pd.read_csv('userMallSet.csv')
    userMallSet_m=userMallSet[userMallSet.mall_id==mallid]
    userMallSet_m['hour']=userMallSet_m['time_stamp'].map(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M").hour)
    userMallSet_m.to_csv('data/userMallSet_'+mallid+'_origin.csv', index=False)

mall=pd.read_csv('mall.csv')

mall.applymap(lambda x: splitDate(x))


