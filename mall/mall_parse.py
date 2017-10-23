import pandas as pd
from sklearn import preprocessing
import pickle


# from sklearn.cross_validation import train_test_split 

# le=preprocessing.LabelEncoder()
# target=le.fit_transform(userMallSet1_m_1021['shop_id'])
# train=preprocessing.scale(userMallSet1_m_1021[['longitude_x','latitude_x']])

# train_X,test_X, train_y, test_y = train_test_split(train,  
#                                                    target,  
#                                                    test_size = 0.2,  
#                                                    random_state = 0)
def get_train_data(mallid):
    userMallSet_m=pd.read_csv('data/userMallSet_'+mallid+'_origin.csv')
    le=preprocessing.LabelEncoder()
    train_y=le.fit_transform(userMallSet_m['shop_id'])
    # print(train_y)
    train_x=userMallSet_m[['longitude_x','latitude_x']]
    return train_x,train_y,le


def get_test_data(mallid):
    evaluationSet=pd.read_csv('evaluation_public.csv')
    evaluationSet_m=evaluationSet[evaluationSet.mall_id==mallid]
    test_x=evaluationSet_m[['longitude','latitude']]
    # print(test_x.count())
    return test_x,evaluationSet_m[['row_id']]

# model = tree.DecisionTreeClassifier()

# model = GradientBoostingClassifier(n_estimators=118)   
# model = SVC(kernel='rbf', probability=True) 


# model = KNeighborsClassifier()      

# print(predict)
# print(test_y)

# from sklearn import metrics 
# precision = metrics.accuracy_score(test_y,predict)
# recall = metrics.recall_score(test_y, predict)    
# print(precision)
# print(le.inverse_transform(predict))
# print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))    
# accuracy = metrics.accuracy_score(test_y, predict)    
# print('accuracy: %.2f%%' % (100 * accuracy))   

def train(train_x,train_y):
    # from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    # from sklearn.ensemble import GradientBoostingClassifier   
    # from sklearn.svm import SVC    
    # from sklearn.neighbors import KNeighborsClassifier 
    model=None
    #    
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model
    
def saveModel(model,mallid):
    with open('model/'+mallid+'.pickle','wb') as f:
        pickle.dump(model,f)

def restoreModel(path):
    with open(path,'rb') as f:
        model=pickle.load(f)
    return model

def run_predict(mallid):
    path='model/'+mallid+'.pickle'
    model=restoreModel(path)
    test_X=get_test_data(mallid)
    predict=model.predict(test_X)

def run_train(mallid):
    train_x,train_y,le=get_train_data(mallid)
    # print(train_x.count())
    # test_model(train_x,train_y)
    model=train(train_x,train_y)
    # saveModel(model,mallid)
    return model,le

def test_model(train_x,train_y):
    from sklearn.cross_validation import train_test_split
    train_X,test_X, train_Y, test_Y = train_test_split(train_x,  
                                                   train_y,  
                                                   test_size = 0.2,  
                                                   random_state = 0)
    model=train(train_X,train_Y)
    predict=model.predict(test_X)
    from sklearn import metrics
    precision = metrics.accuracy_score(test_Y,predict)
    print(precision)

def gogogo(mallid):
    print(mallid)
    model=None
    model,le=run_train(mallid)  
    test_X,data=get_test_data(mallid)
    # print(test_X)
    predict=model.predict(test_X)
    # print(le.inverse_transform(predict))
    data['shop_id']=le.inverse_transform(predict)
    # print(data.count())
    data.to_csv('out/'+mallid+'_result.csv',index=False)

def tetete(mallid):
    print(mallid)
    df=pd.read_csv('out/'+mallid+'_result.csv')

   

mallAll=pd.read_csv('evaluation_public.csv')
# row_id        483931
# print(mallAll.count())
mall=mallAll[['mall_id']].drop_duplicates()
# print(mall)
# gogogo('m_3916')
# for i,row in mall.iterrows():
#     print(row['mall_id'])
#     gogogo(row['mall_id'])

mall.applymap(lambda x: gogogo(x))


#----------------------------------------------------
# list_ = []
# for i,row in mall.iterrows():
#     df=pd.read_csv('out/'+row['mall_id']+'_result.csv')
#     list_.append(df)

# frame = pd.concat(list_)

# print(frame.count())
# frame.to_csv('t_result.csv',index=False)
