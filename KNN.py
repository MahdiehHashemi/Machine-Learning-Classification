import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,jaccard_score
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/classification/teleCust1000t.csv")
# print(df.head())
# print(df.shape)
# print(df["custcat"].value_counts())
df.hist(bins=50)
plt.show()
x=df[[ "region",  "tenure" , "age" , "marital" , "address" , "income" , "ed" , "employ" , "retire" , "gender" , "reside"]]
# print(x)
y=df["custcat"]
print(y.value_counts())
# print(y)
#normalization is necessary as this method works with distances
#new_x=x_data-mean(x)/standard deviation
scaler=StandardScaler().fit(x)
x=scaler.transform(x)  #x=scaler.transform(x.astype(float))
# print(x.shape)
# print(x[0:5])
# random_state fixes the kind of shuffeling that we used (if you wanna show the result to some one and you want the shuffeling to keep fixed)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
#print(x_train.shape,y_train.shape, x_test.shape,y_test.shape)
k=5
m4=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
# print(m4)
p4=m4.predict(x_test)
#diagonal elements are showing 1-1, 2-2, 3-3, 4-4 * the others predict 1-2 , ....
print(confusion_matrix(y_test,p4))
print(y_test[0:5],p4[0:5])
print("accuracy of test: ", accuracy_score(y_test, p4))
print("accuracy of train: ", accuracy_score(y_train, m4.predict(x_train)))
# print("jaccard score: ",jaccard_score(y_test, p4, average='samples') )
ks=10
acc=np.zeros((ks-1))
for n in range (1,ks):
    pn=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    yn_test=pn.predict(x_test)
    acc[n-1]=accuracy_score(y_test,yn_test)
print(acc)
plt.plot(range(1,ks),acc,"g")
plt.show()
x_new_customer=[[1,50,38,1,8,150.000,2,10,1.000,1,3]]
n_c_scaled=scaler.transform(x_new_customer)
print(n_c_scaled)
print(m4.predict(n_c_scaled))
