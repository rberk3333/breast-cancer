#sklearn : It is a library used for ML 
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt 


# (1) Examing the dataset 
cancer=load_breast_cancer()
df=pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target


# (2) Makine ��renmesi Modelinin Se�ilmesi-KNN S�n�fland�r�c�
# (3) Training the model
X = cancer.data #features
y = cancer.target #targets

# train test split
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size = 0.3, random_state = 42)

# �l�eklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors= 3) # model olu�turma kom�u parametresini unutma*******
knn.fit(X_train, y_train) # you need to add features and targets as x and y /fit fonku verimizi samples targetleri kullanarak knn algoritmas�n� e�itir

# (4)Sonu�lar�n De�erlendirilmesi : test
y_pred= knn.predict(X_test) #predict X e ihtiya� duyar ve bir parametre al�r peki bu x ne (sampledir)

#pred i�in traine de X verdik Test e de X verdik bu uygun de�il

accuracy = accuracy_score( y_test, y_pred)
print( "Do�ruluk:", accuracy )

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(conf_matrix)

# (5) hiperparametre ayarlamas�
"""
   KNN: hYPERPARAMETER =K
   K: 1, 2, 3,....N
   Accuracy: %A,%B,%C ...
   
"""
accuracy_values = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)
    

plt.figure()
plt.plot(k_values, accuracy_values, marker = "o" , linestyle = "-")
plt.xlabel("K degeri")
plt.ylabel("Dogruluk")
plt.xticks(k_values)
plt.grid(True)
  
 #%%
import numpy as np
import matplotlib.pyplot as plt

   
X = np.sort(5 * np.random.rand(40,1),axis = 0) # uniform
y = np.sin(X) #target
plt.scatter(X,y)

# add noise
y[:5] += 1 * (0.5 - np.random.rand(5))

#plt.scatter(X,y)
T = np.linspace(0,5,500)[:,np.newaxis]

for weight in ["uniform", "distance"]:
    
    knn = KNeighboorsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fir(X,y).predict(T)

    plt.figure()
    plt.scatter(X,y, color = "green",label = "data")
    plt.plot(T, y_pred, color = "blue", label= "prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights = {}".format(weight))
    
plt.tight_layout()
plt.show()
    


