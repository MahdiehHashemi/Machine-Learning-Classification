import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from io import StringIO
import pydotplus
import matplotlib.image as mpimg

df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/classification/drug200.csv", delimiter=",")
#print(df.head(5))
#print(df.shape)
x=df[["Age", "Sex","BP", "Cholesterol","Na_to_K"]].values
#print(x[0:5])
y=df["Drug"]
print(y.value_counts())
# l_drug=LabelEncoder()
# l_drug.fit(["drugY","drugX","drugA","drugC","drugB"])
# y=l_drug.transform(y)
# print(y)
l_sex=LabelEncoder()
l_sex.fit(['F','M'])
x[:,1]=l_sex.transform(x[:,1])
l_BP=LabelEncoder()
l_BP.fit(["LOW","NORMAL","HIGH"])
x[:,2]=l_BP.transform(x[:,2])
l_cho=LabelEncoder()
l_cho.fit(["NORMAL","HIGH"])
x[:,3]=l_cho.transform(x[:,3])
#print(x[0:5])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=3)
drug=DecisionTreeClassifier(criterion="entropy",max_depth=4) # with maximum 4 layers of tree (subbranches)
drug.fit(x_train,y_train)
#drug_predict=drug.predict(x_test)
predicted_drug=drug.predict(x_test)
print(y_test[0:5])
print(predicted_drug[0:5])
print(accuracy_score(y_test,predicted_drug))
print(accuracy_score(y_train,drug.predict(x_train)))
# dot_data=StringIO()
# filename="drugtree.png"
# dfn=df.columns[0:5]
# out=export_graphviz(drug,feature_names=dfn, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png(filename)
# img = mpimg.imread(filename)
# plt.figure(figsize=(100, 200))
# plt.imshow(img,interpolation='nearest')