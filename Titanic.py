#!/usr/bin/env python
# coding: utf-8

# In[332]:


import seaborn as sns
import pandas as pd
import numpy as np
import numpy as np 
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error,roc_curve,roc_auc_score, r2_score,classification_report, confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn.svm import SVC


# In[400]:


from warnings import filterwarnings
filterwarnings('ignore',category = DeprecationWarning)
filterwarnings('ignore',category = FutureWarning)


# In[333]:


titanic = pd.read_csv("titanic.csv")


# In[334]:


df = titanic.copy()


# In[335]:


df.head()


# In[336]:


df = df.drop(["embark_town","who","adult_male","deck","alive","alone"], axis = 1)


# In[337]:


df.head()


# In[261]:


df.info()


# In[52]:


df["age"].mean()


# In[12]:


df.shape


# In[13]:


df.isnull().sum()


# In[14]:


df.dtypes


# In[262]:


df.describe().T


# In[32]:


a=sns.barplot(x="sex", y="survived", hue="sex", data=df);
a.set_title("Cinsiyete Göre Hayatta Kalma Dağılımı");


# In[28]:


a=sns.barplot(x="sex", y="survived", hue="pclass", data=df);
a.set_title("Cinsiyete Göre Survived Dağılımı");


# In[46]:


b = sns.catplot(x="pclass", y="fare", kind="bar", hue="pclass", col="sex", orient="v", data=df);
b.fig.suptitle("Cinsiyete Göre Bilet Sınıf Dağılımı");


# In[50]:


a=sns.barplot(x="sex", y="survived", hue="parch", data=df);
a.set_title("Cinsiyete Göre Ebeveyn/Çocuk Dağılımı");


# In[51]:


df.boxplot(column = "fare")


# In[60]:


a = sns.catplot(x = "class", y = "fare", data = df)
a.fig.suptitle("Bilet Sınıfına Göre Ücret Dağılımı");


# In[58]:


a = sns.boxplot(x  = df["fare"]);
a.set_title("Bilet Fiyatı Dağılımı");


# In[60]:


df["fare"].hist()


# In[59]:


df["age"].hist()


# In[315]:


a = sns.boxplot(x  = df["age"]);
a.set_title("Yaş Dağılımı");


# In[12]:


df["age"] = pd.cut(df["age"].astype(float),5)


# In[13]:


df[["age","survived"]].groupby("age")["survived"].mean().sort_values().plot(kind = "bar");


# In[10]:


df["fare"] = pd.cut(df["fare"].astype(float),5)


# In[11]:


df[["fare","survived"]].groupby("fare")["survived"].mean().sort_values().plot(kind = "bar");


# In[14]:


df.isnull().sum()


# In[170]:


str(df["age"].isnull().sum()  / len(df["age"]) * 100)


# In[179]:


df.info()


# In[191]:


display(df.groupby("pclass")["age"].median())


# In[192]:


display(df.groupby(["pclass","sex"])["age"].median())


# In[197]:


display(df.groupby(["pclass","sex","sibsp"])["age"].median())


# In[201]:


df.groupby(["pclass","sex","parch"])["age"].median()


# In[338]:


df["age"] = df.groupby(["pclass","sex"])["age"].apply(lambda x : x.fillna(x.median()))


# In[339]:


df.info()


# In[55]:


df["age"].mean()


# In[41]:


df["age"].describe().T


# In[190]:


df.loc[df["embarked"].isnull()]


# In[340]:


df.loc[(df["pclass"] == 1) & (df["sex"] == "female") & (df["sibsp"]) == 0 & (df["parch"] == 0) & (df["age"].mean() == 50)]["embarked"].value_counts()


# In[341]:


df.loc[df["embarked"].isnull(), "embarked"] = "S"


# In[342]:


df.isnull().sum()


# **- Aykırı Değer Problemini Çözme**

# In[343]:


df["age"].describe().T


# In[344]:


df_age = df["age"]  #yaş değişkenini seç
df_age.head()


# In[345]:


#eşik değer belirleme
Q1 = df_age.quantile(0.25) #ilk çeyreklik
Q3 = df_age.quantile(0.75) #üçüncü çeyreklik
IQR = Q3-Q1 # veri interquartile, üçüncü çeyrek farkı birinci çeyrek farkı


# In[318]:


Q1


# In[319]:


Q3


# In[320]:


IQR


# **- Alt Sınır ve Üst Sınır Belirleme**

# In[346]:


alt_sinir = Q1- 1.5*IQR 
ust_sinir = Q3 + 1.5*IQR


# In[347]:


print("Alt Sınır : "  + str(alt_sinir) + "\nÜst Sınır : " + str(ust_sinir))


# In[353]:


import numpy as np
from scipy.stats import iqr

def outliers(df, factor=1.5):
    for i in range(0, len(df)):
        if df["age"][i] > ust_sinir:
            df["age"][i] = ust_sinir
    return df["age"]

outlier = outliers(df)

df.age.replace(outlier, inplace = True)


# In[326]:


aykiri_degerler = (df_age > ust_sinir)


# In[252]:


df['age'] = df['age'].replace(aykiri_degerler, np.nan).interpolate()


# In[327]:


df_age[aykiri_degerler].index


# In[203]:


df[aykiri_degerler] = ust_sinir


# In[355]:


df.head()


# In[354]:


a = sns.boxplot(x  = df["age"]);
a.set_title("Yaş Dağılımı");


# In[356]:


df_fare = df["fare"]


# In[357]:


#eşik değer belirleme
Q1 = df_fare.quantile(0.25) #ilk çeyreklik
Q3 = df_fare.quantile(0.75) #üçüncü çeyreklik
IQR = Q3-Q1 # veri interquartile, üçüncü çeyrek farkı birinci çeyrek farkı


# In[358]:


alt_sinir = Q1- 1.5*IQR 
ust_sinir = Q3 + 1.5*IQR


# In[359]:


Q1


# In[360]:


Q3


# In[361]:


IQR


# In[363]:


import numpy as np
from scipy.stats import iqr

def outliers(df, factor=1.5):
    for i in range(0, len(df)):
        if df["fare"][i] > ust_sinir:
            df["fare"][i] = ust_sinir
    return df["fare"]

outlier = outliers(df)

df.fare.replace(outlier, inplace = True)


# In[ ]:





# In[174]:


aykiri_degerler = (df_fare > ust_sinir)


# In[175]:


df_fare[aykiri_degerler].head()


# In[176]:


df[aykiri_degerler] = ust_sinir


# In[364]:


a = sns.boxplot(x  = df["fare"]);
a.set_title("Bilet Fiyatı Dağılımı");


# **ONE HOT ENCODING DÖNÜŞÜMÜ**

# In[365]:


df.head()


# In[366]:


df = pd.get_dummies(df, columns = ["sex","embarked","pclass","class"], prefix = ["sex","embarked","pclass","class"])


# In[368]:


df.head()


# In[383]:


y = df["survived"]
X = df.drop(["survived"], axis = 1)


# **FEATURE ENGINEERING**

# In[371]:


df["Family"] = df["sibsp"] + df["parch"] + 1


# In[296]:





# In[372]:


df.head()


# In[373]:


df[["Family","survived"]].groupby("Family")["survived"].mean().sort_values().plot(kind = "bar");


# In[139]:


df.survived = df.survived.astype(int)


# In[384]:


#Model Tuning
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.20,
                                                    random_state = 42) 


# **MODELLEME VE TAHMİN**

# **- Lojistik Regresyon**

# In[385]:


loj_model = LogisticRegression(solver = "liblinear").fit(X,y)


# In[468]:


loj_model


# In[467]:


get_ipython().run_line_magic('pinfo', 'loj_model')


# In[386]:


y_pred = loj_model.predict(X)


# In[387]:


accuracy_score(y, y_pred)


# In[388]:


logit_roc_auc = roc_auc_score(y, loj_model.predict(X))
fpr, tpr, thresholds = roc_curve(y,loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr,tpr,label = "AUC (Area %0.2f)" % logit_roc_auc)
plt.plot([0,1],[0,1],"r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc = "lower right")
plt.savefig("Log_ROC")
plt.show()


# **- K En Yakın Komşu**

# In[389]:


knn_model = KNeighborsClassifier().fit(X_train,y_train)


# In[390]:


y_pred = knn_model.predict(X_test)


# In[391]:


knn_model


# In[392]:


accuracy_score(y_test,y_pred)


# **- Destek Vektör Makineleri**

# In[393]:


svm_model = SVC(kernel = "linear").fit(X_train,y_train)


# In[316]:


svm_model


# In[394]:


y_pred = svm_model.predict(X_test)


# In[395]:


accuracy_score(y_test,y_pred)


# **- Yapay Sinir Ağları**

# In[401]:


mlpc_model = MLPClassifier().fit(X_train,y_train)


# In[402]:


y_pred = mlpc_model.predict(X_test)


# In[403]:


accuracy_score(y_test,y_pred)


# **- Classification and Regression Trees (CART)**

# In[630]:


cart_model = DecisionTreeClassifier().fit(X_train,y_train)


# In[631]:


cart_model


# In[632]:


y_pred = cart_model.predict(X_test)


# In[633]:


accuracy_score(y_test,y_pred)


# **- Random Forests**

# In[408]:


rf_model = RandomForestClassifier().fit(X_train,y_train)


# In[409]:


rf_model


# In[410]:


y_pred = rf_model.predict(X_test)


# In[411]:


accuracy_score(y_test,y_pred)


# **- Gradient Boosting Machines (GBM)**

# In[412]:


gbm_model = GradientBoostingClassifier().fit(X_train,y_train)


# In[413]:


gbm_model


# In[414]:


y_pred = gbm_model.predict(X_test)


# In[415]:


accuracy_score(y_test,y_pred)


# **- XGBoost**

# In[416]:


get_ipython().system('pip install xgboost')


# In[417]:


from xgboost import XGBClassifier


# In[418]:


xgb_model = XGBClassifier().fit(X_train,y_train)


# In[419]:


y_pred = xgb_model.predict(X_test)


# In[420]:


accuracy_score(y_test,y_pred)


# **- Light GBM**

# In[421]:


get_ipython().system('pip install lightgbm')


# In[422]:


get_ipython().system('conda install -c conda-forge lightgbm')


# In[423]:


from lightgbm import LGBMClassifier


# In[424]:


lgbm_model = LGBMClassifier().fit(X_train,y_train)


# In[425]:


y_pred = lgbm_model.predict(X_test)


# In[426]:


accuracy_score(y_test,y_pred)


# **- Cat Boost**

# In[430]:


get_ipython().system('pip install catboost')


# In[428]:


from catboost import CatBoostClassifier


# In[431]:


catb_model = CatBoostClassifier().fit(X_train,y_train,verbose = False)


# In[432]:


y_pred = catb_model.predict(X_test)


# In[433]:


accuracy_score(y_test,y_pred)


# **- TÜM MODELLERİN KARŞILAŞTIRILMASI**

# In[461]:


modeller = [loj_model,
           knn_model,
           svm_model,
           mlpc_model,
           cart_model,
           rf_model,
           gbm_model,
           xgb_model,
           lgbm_model,
           catb_model]

sonuc = []
sonuclar = pd.DataFrame(columns = ["Modeller", "Accuracy"])

for model in modeller:
    isimler = model.__class__.__name__
    y_pred  = model.predict(X_test)
    dogruluk = accuracy_score(y_test,y_pred)
    sonuc = pd.DataFrame([[isimler, dogruluk*100]], columns = ["Modeller","Accuracy"])
    sonuclar = sonuclar.append(sonuc)
    
sonuclar = sonuclar.sort_values(["Accuracy"], ascending = False)


# In[465]:


sns.barplot(x = "Accuracy", y = "Modeller" , data = sonuclar)
plt.xlabel("Accuracy %")
plt.title("Modellerin Dogruluk Oranları")


# **MODEL TUNING**

# In[469]:


loj_model


# In[ ]:


#Öncelikle lojistik regresyon modelini tune edeceğiz. 


# In[521]:


loj_params = {"solver": ["lbfgs", "sag","saga","liblinear"],
             "C":[1, 10, 100, 1000, 10000],
             "max_iter": [1000,2000,5000,10000,20000]}


# In[494]:


loj = LogisticRegression()


# In[522]:


loj_cv_model = GridSearchCV(loj,loj_params,cv = 10, n_jobs = -1, verbose = 2).fit(X_train,y_train)


# In[523]:


loj_cv_model.best_params_


# In[524]:


loj_tuned = LogisticRegression(solver = "lbfgs",C = 1,max_iter = 1000).fit(X,y)


# In[525]:


y_pred = loj_tuned.predict(X_test)


# In[526]:


accuracy_score(y_test,y_pred)


# In[ ]:


#Cart model için optimizasyon


# In[527]:


cart = DecisionTreeClassifier()


# In[579]:


get_ipython().run_line_magic('pinfo', 'cart')


# In[581]:


cart_params = {"max_depth": [1,2,3,4,5,6,7,8,9,10],
              "min_samples_split": [1,2,3,4,5,10,20,50],
              "max_features": [0.2,0.4,0.6,0.8],
              "min_samples_leaf":[1,2,3,4,5,6,7,8]}


# In[582]:


cart_cv_model = GridSearchCV(cart, cart_params, cv = 10, verbose = 2, n_jobs = -1).fit(X_train,y_train)


# In[583]:


cart_cv_model.best_params_


# In[627]:


cart_tuned = DecisionTreeClassifier(max_depth = 3, min_samples_split = 50, max_features = 0.4, min_samples_leaf = 3).fit(X_train,y_train)


# In[628]:


y_pred = cart_tuned.predict(X_test)


# In[629]:


accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[637]:


feature_imp = pd.Series(rf_model.feature_importances_,
                       index = X_train.columns).sort_values(ascending = False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel("Değişken Önem Skorları")
plt.ylabel("Değişkenler")
plt.title("Değişken Önem Düzeyleri")
plt.show()


# In[ ]:





# In[134]:


rf_model = RandomForestClassifier().fit(X_train,y_train)


# In[208]:


rf_model


# In[231]:


y_pred = rf_model.predict(X_test)


# In[232]:


accuracy_score(y_test,y_pred)


# In[211]:


rf = RandomForestClassifier()


# In[212]:


rf_params = {"n_estimators": [100,200,500,1000],
            "max_features":[3,5,7,8],
            "min_samples_split":[2,5,10,20]}


# In[213]:


rf_cv_model = GridSearchCV(rf,rf_params,cv = 10, verbose=2,n_jobs=-1).fit(X_train,y_train)


# In[214]:


rf_cv_model.best_params_


# In[227]:


rf_tuned = RandomForestClassifier(max_features = 7, min_samples_split = 20,
                                 n_estimators = 200).fit(X_train,y_train)


# In[228]:


y_pred = rf_tuned.predict(X_test)


# In[229]:


accuracy_score(y_test,y_pred)


# In[233]:


feature_imp = pd.Series(rf_tuned.feature_importances_,
                       index = X_train.columns).sort_values(ascending = False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel("Değişken Önem Skorları")
plt.ylabel("Değişkenler")
plt.title("Değişken Önem Düzeyleri")
plt.show()


# In[296]:


#Cat Boost 


# In[297]:


get_ipython().system('pip install catboost')


# In[ ]:





# In[126]:


from catboost import CatBoostClassifier


# In[285]:


catb_model


# In[286]:


catb_model = CatBoostClassifier().fit(X_train,y_train,verbose = False)


# In[287]:


y_pred = catb_model.predict(X_test)


# In[288]:


accuracy_score(y_test,y_pred)


# In[303]:


#Model Tuning


# In[237]:


catb = CatBoostClassifier()


# In[ ]:





# In[238]:


catb_params = {"iterations":[200,500,1000],
              "learning_rate": [0.01, 0.03, 0.1],
              "depth":[4,5,8]}


# In[239]:


catb_cv_model = GridSearchCV(catb, catb_params, cv = 5, verbose = 2, n_jobs = -1).fit(X_train,y_train)


# In[240]:


catb_cv_model.best_params_


# In[282]:


catb_tuned = CatBoostClassifier(iterations = 1000,
                               learning_rate = 0.01,
                               depth = 4).fit(X_train,y_train)


# In[283]:


y_pred = catb_tuned.predict(X_test)


# In[284]:


accuracy_score(y_test,y_pred)


# In[276]:


rf_tuned2 = RandomForestClassifier(criterion = "gini",
                                  max_depth = 7,
                                   min_samples_split = 6,
                                  min_samples_leaf = 6,
                                  max_features = "auto",
                                  oob_score = True,
                                  random_state = 42,
                                  n_jobs = -1,
                                  verbose = 1,
                                 n_estimators = 1750).fit(X_train,y_train)


# In[277]:


y_pred = catb_tuned.predict(X_test)


# In[278]:


accuracy_score(y_test,y_pred)


# In[289]:


feature_imp = pd.Series(gbm_tuned.feature_importances_,
                       index = X_train.columns).sort_values(ascending = False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel("Değişken Önem Skorları")
plt.ylabel("Değişkenler")
plt.title("Değişken Önem Düzeyleri")
plt.show()


# In[ ]:





# In[273]:


gbm_model = GradientBoostingClassifier().fit(X_train,y_train)


# In[251]:


gbm_model


# In[274]:


y_pred = gbm_model.predict(X_test)


# In[275]:


accuracy_score(y_test,y_pred)


# In[ ]:





# In[254]:


gbm = GradientBoostingClassifier()


# In[255]:


gbm


# In[256]:


gbm_params = {"learning_rate" : [0.1, 0.01, 0.001, 0.05],
             "n_estimators" : [100, 300, 500, 1000],
             "max_depth": [2,3,5,8]}


# In[257]:


gbm_cv_model = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train,y_train)


# In[258]:


gbm_cv_model.best_params_


# In[267]:


gbm_tuned = GradientBoostingClassifier(learning_rate = 0.01,
                                       n_estimators = 3,
                                       max_depth = 500).fit(X_train,y_train)


# In[268]:


y_pred = gbm_model.predict(X_test)


# In[269]:


accuracy_score(y_test,y_pred)


# In[ ]:




