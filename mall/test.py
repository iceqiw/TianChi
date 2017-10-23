import pandas as pd
from sklearn import preprocessing
import pickle

n_estimator = 10

def get_train_data(mallid):
    userMallSet_m=pd.read_csv('data/userMallSet_'+mallid+'_origin.csv')
    le=preprocessing.LabelEncoder()
    train_y=le.fit_transform(userMallSet_m['shop_id'])
    # print(train_y)
    train_x=userMallSet_m[['longitude_x','latitude_x']]
    return train_x,train_y


def train(train_x,train_y):
    # from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    # from sklearn.ensemble import GradientBoostingClassifier   
    # from sklearn.svm import SVC    
    # from sklearn.neighbors import KNeighborsClassifier 
 
    model = RandomForestClassifier(n_estimators=n_estimator)
    model.fit(train_x, train_y)
    return model

def test_model(mallid):
    train_x,train_y=get_train_data(mallid)
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



test_model('m_615')