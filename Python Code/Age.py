import os

import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
data = pd.read_csv("./China__Korea_total_data.csv")

# data.Country = data.Country.replace("China", 0)
# data.Country = data.Country.replace("Korea", 1)

data.Age = data.Age.replace([17,18,19,20,21], "Low")
data.Age = data.Age.replace([22,23,24,25,26], "Mid")
data.Age = data.Age.replace([27,28,29,32,33,38,45,50,54,82], "High")
data.Age = data.Age.replace("Low",0)
data.Age = data.Age.replace("Mid",1)
data.Age = data.Age.replace("High",2)

X = np.array(pd.DataFrame(data, columns = ['PL1', 'PL2', 'PL3', 'Pl4',
                                           'F1', 'F2','F3', 'F4',
                                           'PR1', 'PR2', 'PR3', 'PR4',
                                           'PE1', 'PE2', 'PE3', 'PE4',
                                           'D1', 'D2', 'D3',
                                           'E1', 'E2', 'E3', 'E4']))
y = np.array(pd.DataFrame(data, columns = ['Age']))

X_train, X_test, y_train, y_test = train_test_split(X,y)


# dt_clf = DecisionTreeClassifier(max_depth = 7, random_state = 1)
dt_clf =  DecisionTreeClassifier(max_depth=15, random_state=0)
dt_clf = dt_clf.fit(X_train, y_train)
dt_prediction = dt_clf.predict(X_test)
feature_names = data.columns.tolist()
feature_names = feature_names[3:26]
target_name = np.array(["Low","Mid","High"])

print("훈련 세트 정확도 : {:.3f}".format(dt_clf.score(X_train,y_train)))
print("테스트 세트 정확도 : {:.3f}".format(dt_clf.score(X_test,y_test)))
print("특성 중요도 : \n{}".format(dt_clf.feature_importances_))

dt_dot_data = tree.export_graphviz(dt_clf, out_file = 'tree.dot',
                                    feature_names = feature_names,
                                    class_names = target_name,
                                    filled = True, rounded = True,
                                    rotate=True,
                                    special_characters = True)

with open('tree.dot') as file_reader:
    dot_graph = file_reader.read()

dot = graphviz.Source(dt_dot_data)
dot = graphviz.Source(dot_graph)
dot.render(filename='Age_15.png')
