#support vector mechine
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.colors as cplt

df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/classification/cell_samples.csv", delimiter=",")
print(df['Class'].value_counts())
col=cplt.ListedColormap(['yellow','green'])
plt.scatter(df['Clump'],df['UnifSize'],c=df['Class'],cmap=col)
plt.show()
print(df.dtypes)
print(df["BareNuc"].value_counts()) ## ? is something unwanted in here
n_df= df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()] 
#notnull. Detect non-missing values for an array-like object. This function takes a scalar or array-like object and indicates whether values are valid (not missing, which is NaN in numeric arrays, None or NaN in object arrays, NaT in datetimelike).
n_df['BareNuc']=(n_df['BareNuc']).astype('int') 
print(n_df.dtypes)
#print(n_df['BareNuc'].value_counts())
# print(df.size)# =7689
# print(n_df.size) ##=7513=7689-176: 176=16(number of ?)*11(number of columns)
x = n_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
y = n_df['Class']
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
model=svm.SVC(kernel='rbf') #radial basic function
model.fit(X_train, y_train)
y_test_p=model.predict(X_test)
cnf_matrix=confusion_matrix(y_test, y_test_p,labels=[2,4])
print(cnf_matrix)
print(classification_report(y_test,y_test_p))
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap="Blues"):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['type=2','type=4'],  title='Confusion matrix')
plt.show()