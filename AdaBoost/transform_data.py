from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,make_hastie_10_2
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# data = pd.read_csv('./dataset/uci-credit-card.csv')
data = pd.read_csv('./dataset/letter-recognition.csv')

Y_data = data['target'].values
X_data = data.drop('target',axis=1).values

X_train, X_test, Y_train, Y_test = \
    train_test_split(X_data,Y_data,test_size=0.33,random_state=42)

X_train_1, X_train_2, Y_train_1, Y_train_2 = \
    train_test_split(X_train,Y_train,test_size=0.5,random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train_1,Y_train_1)
print("Accuracy: ", accuracy_score(Y_test,clf.predict(X_test)))

Y_pred_2 = clf.predict(X_train_2)
clf2 = DecisionTreeClassifier()
clf2.fit(X_train_2,Y_train_2)
print("Accuracy: ", accuracy_score(Y_test,clf2.predict(X_test)))

rf = RandomForestClassifier()
rf.fit(X_train_2,Y_pred_2)
print("Accuracy: ", accuracy_score(Y_test,rf.predict(X_test)))

print("Acc:",accuracy_score(Y_train_2,Y_pred_2))