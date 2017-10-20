import pandas as pd

merchantSet=pd.read_csv('ccf_first_round_shop_info.csv')
# print(merchantSet.sort_values(by=('mall_id')).head(10))
merchantSet_mallid=merchantSet[['mall_id']].drop_duplicates()

merchantSet_mallid.to_csv('mall.csv', index=False)

# print(merchantSet_g_mallid.mean())

userSet=pd.read_csv('ccf_first_round_user_shop_behavior.csv')
# print(userSet.sort_values(by=('shop_id')).head(10))

userMallSet=pd.merge(userSet,merchantSet,how='left',on=['shop_id'])

userMallSet1=userMallSet[['user_id','shop_id','mall_id','category_id','price','longitude_x','latitude_x','time_stamp','wifi_infos']]
# print(userMallSet1.sort_values(by=('mall_id')).head(10))
userMallSet1.to_csv('userMallSet.csv', index=False)

