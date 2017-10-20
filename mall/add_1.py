import pandas as pd

userMallSet=pd.read_csv('userMallSet.csv')

userCategorySet= userMallSet.groupby(['user_id','category_id'])

userCategorySet.to_csv('userCategorySet.csv', index=False)