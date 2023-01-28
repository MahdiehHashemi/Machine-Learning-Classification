import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, jaccard_score,log_loss

df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/classification/ChurnData.csv", delimiter=",")
x=np.asanyarray(df[["tenure","age","address","income","ed","employ","equip"]])
y=np.asanyarray(df["churn"].astype(int))
scaler=StandardScaler().fit(x)
x=scaler.transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=4)
LR=LogisticRegression(C=0.01,solver="liblinear").fit(x_train,y_train) #c, overfit ra tanzim mikone 
yp=LR.predict(x_test)
print(y_test,yp)
e=jaccard_score(y_test,yp,pos_label=0) # jaccard baraye yek meghdar faghat kar mikonad ke dar position_label moshakhas mishavad
print(e)#oonhaii ke mikhan beran ro ba in deghat pishbini mikone 
ee=log_loss(yp,y_test)
print(ee)
from sklearn.metrics import  confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yp, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yp, labels=[1,0])
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()