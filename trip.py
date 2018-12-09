import numpy as np
import pandas as pd
from IPython.display import display
data = pd.read_csv('Potential datasets for recruitment (Trip).csv')
display(data.describe(include=[np.number]))
display(data.describe(exclude=[np.number]))
#pd.plotting.scatter_matrix(data,figsize=(10,10),diagonal='kde',s=40,alpha=0.5,marker='*',color='green');

categorical = list(data.select_dtypes(include=['object']).columns.values)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
['Dec-Feb' 'Mar-May' 'Jun-Aug' 'Sep-Nov']
data['Period of stay'] = data['Period of stay'].map({'Dec-Feb':'winter', 'Mar-May':'spring', 'Jun-Aug' :'summer','Sep-Nov':'autumn'})

for i in range(0, len(categorical)):
  #  data[categorical[i]] = le.fit_transform(data[categorical[i]])
    print(data[categorical[i]].unique())
for i in range(0, len(categorical)):
    data[categorical[i]] = le.fit_transform(data[categorical[i]])
    
print(data.head())
from sklearn.model_selection import train_test_split
X= data.drop(['Score'], axis=1) ## remove score label from data
y = data['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

rfe = RFE(estimator = Ridge(), n_features_to_select = 12)
rfe.fit(X_train, y_train)
feature_list = pd.DataFrame({'col':list(X_train.columns.values),'sel':list(rfe.support_ *1)})
print("*Most contributing features in Score*")

print(feature_list[feature_list.sel==1].col.values)
X_sel = pd.DataFrame(X_train, columns=(feature_list[feature_list.sel==1].col.values))
X_sel_t = pd.DataFrame(X_test, columns=(feature_list[feature_list.sel==1].col.values))
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_sel, y_train)
p = (list(clf.predict(X_sel_t)))
print(clf.score(X_sel_t, y_test))
