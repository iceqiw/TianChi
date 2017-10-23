
import pandas as pd

#----------------------------------------------------

mallAll=pd.read_csv('evaluation_public.csv')
mall=mallAll[['mall_id']].drop_duplicates()
list_ = []
for i,row in mall.iterrows():
    df=pd.read_csv('out/'+row['mall_id']+'_result.csv')
    list_.append(df)

frame = pd.concat(list_)

print(frame.count())
frame.to_csv('t_result.csv',index=False)