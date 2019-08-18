#!/usr/bin/env python
# coding: utf-8

# In[115]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.metrics
from sklearn import ensemble
from scipy import stats as st 
from sklearn import linear_model,datasets
from sklearn.metrics import mean_squared_error


# In[212]:


from xgboost import XGBRegressor
from hyperopt import hp,fmin,tpe
from sklearn.model_selection import cross_val_score, KFold


# In[ ]:





# In[211]:


get_ipython().system('pip install hyperopt')


# In[ ]:





# In[209]:


get_ipython().system('pip install xgboost')


# In[ ]:





# In[ ]:





# In[198]:


data=pd.read_csv(r"D:\ml\blackfridayt\train.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[199]:


data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[118]:


# data cleanning


# In[119]:


missing_values = data.isnull().sum().sort_values(ascending = False)


# In[120]:


missing_values


# In[121]:


missing_values = missing_values[missing_values > 0]/data.shape[0]
print(f'{missing_values *100} %')


# In[122]:


"""believe that the NaN values for Product_Category_2 and Product_Categrory_3 would mean that the concerned person did not buy the products from these categories.

Hence, I believe that it would be safe to replace them with 0."""


# In[123]:


data = data.fillna(0)


# In[124]:


missing_values = data.isnull().sum().sort_values(ascending = False)
missing_values = missing_values[missing_values > 0]/data.shape[0]
print(f'{missing_values *100} %')


# In[125]:


data.dtypes


# In[126]:


# So, the available datatypes are : int64, float64 and objects. We will leave the numeric datatypes alone and focus on object datatypes as the cannot be directly fen into a Machine Learning Model


# In[200]:


gender = np.unique(data['Gender'])
gender


# In[ ]:





# In[ ]:





# In[128]:


# So, we do not have any 'Other' gender type. I will create a fuction and map M=1 and F=0. No sexism intended.
def map_gender(gender):
    if gender == 'M':
        return 1
    else:
        return 0
data['Gender'] = data['Gender'].apply(map_gender)


# In[195]:


age = np.unique(data['Age'])
age


# In[ ]:





# In[130]:


def map_age(age):
    if age == '0-17':
        return 0
    elif age == '18-25':
        return 1
    elif age == '26-35':
        return 2
    elif age == '36-45':
        return 3
    elif age == '46-50':
        return 4
    elif age == '51-55':
        return 5
    else:
        return 6
data['Age']=data['Age'].apply(map_age)    


# In[131]:


city_category = np.unique(data['City_Category'])
city_category


# In[132]:


def map_city_categories(city_category):
    if city_category == 'A':
        return 2
    elif city_category == 'B':
        return 1
    else:
        return 0
data['City_Category'] = data['City_Category'].apply(map_city_categories)


# In[133]:


city_stay = np.unique(data['Stay_In_Current_City_Years'])
city_stay


# In[134]:


def map_stay(stay):
        if stay == '4+':
            return 4
        else:
            return int(stay)
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].apply(map_stay)       


# In[135]:


cols = ['User_ID','Product_ID']
data.drop(cols, inplace = True, axis =1)


# In[136]:


data.describe()


# In[137]:


data.head()


# In[138]:


# eda
data[['Gender','Purchase']].groupby('Gender').mean().plot.bar()
sns.barplot('Gender', 'Purchase', data = data)
plt.show()


# In[201]:


data[['Age','Purchase']].groupby('Age').mean().plot.bar()
sns.barplot('Age', 'Purchase', data = data)
plt.show()


# In[ ]:





# In[140]:


sns.boxplot('Age','Purchase', data = data)
plt.show()


# In[141]:


# Not much of a deciation there. We can say that no matter what age group you belong to, you are gonna make full use of your purchasing power on a Black Friday


# In[142]:


data[['City_Category','Purchase']].groupby('City_Category').mean().plot.bar()
sns.barplot('City_Category', 'Purchase', data = data)
plt.show()


# In[143]:


# Okay so, the people belonging to category 0 tend to spend a little more. These may be the more developed cities that we are talking about here.


# In[144]:


corrmat = data.corr()
fig,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[145]:


mean_cat_1 = data['Product_Category_1'].mean()
mean_cat_2 = data['Product_Category_2'].mean()
mean_cat_3= data['Product_Category_3'].mean()
print(f"PC1: {mean_cat_1} \n PC2: {mean_cat_2} \n PC3 : {mean_cat_3}")


# In[146]:


print(data.skew())


# In[159]:


def skewness_check(data):
    # Find the skewness in the dataset
    skew_value = list(st.skew(data))
    skew_string = []
    # Looping through the skew value to find the Skew category
    for skew in skew_value:
        if skew >= -.5 and skew <= .5:
            skew_string.append("Light Skewed")
        elif skew <= -.5 and skew >= -1 and skew <= .5 and skew >= 1:
            skew_string.append("Moderately Skewed")
        else:
            skew_string.append("Heavily Skewed")
    # Ctreating data frame
    skew_df = pd.DataFrame({'Column': data.columns, 'Skewness': skew_value, 'Skew Category': skew_string})
    return skew_df

# Skewness for Red Wine
skewness_check(data.iloc[:, :-1])


# In[173]:


# boxcox Transformation
data[["Gender"]] += 0.1
data[["Age"]] += 0.1
data[["Occupation"]] += 0.1
data[["City_Category"]] += 0.1
data[["Stay_In_Current_City_Years"]]+=.1
data[["Marital_Status"]]+=.1
data[["Product_Category_2"]] += 0.1

data[["Product_Category_3"]] += 0.1
def boxcox_trans(data):
    for i in range(data.shape[1]):
        data.iloc[:, i], _ = st.boxcox(data.iloc[:, i])
    return data
# Subset the predcitors
red_trans = data.copy(deep = True)
red_trans.iloc[:, :-1] = boxcox_trans(red_trans.iloc[:, :-1])
skewness_check(red_trans.iloc[:, :-1])


# In[ ]:





# In[174]:


def subplot_hist(data, row = 4, column = 3, title = "Subplots", height = 20, width = 19):
    # Create a figure instance, and the two subplots
    fig = plt.figure(figsize = (width, height))
    fig.suptitle(title, fontsize=25, y = 0.93)
    # Run loop over the all the variables
    for i in range(data.shape[1]):
        # Create the axis line
        ax = fig.add_subplot(row, column, i + 1)
        fig.subplots_adjust(hspace = .5)
        # Create histogram for each variable
        sns.distplot(data.iloc[:, i], ax=ax)

    # Show the plot
    plt.show()
    
subplot_hist(data.iloc[:, :-1], row = 4, column = 3, title = "Histogram of the Black Predictors")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[175]:


age = np.unique(data['Age'])
age


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[176]:


city_stay = np.unique(data['Stay_In_Current_City_Years'])
city_stay


# In[ ]:





# In[177]:


city_category = np.unique(data['City_Category'])
city_category


# In[ ]:





# In[178]:


gender = np.unique(data['Gender'])
gender


# In[ ]:





# In[179]:


Y = data["Purchase"]


# In[ ]:





# In[180]:


from sklearn.preprocessing import StandardScaler
SS = StandardScaler()


# In[ ]:





# In[181]:


X = data.drop(["Purchase"], axis=1)
X


# In[184]:


X.iloc[:, :-3],


# In[206]:


from sklearn.model_selection import train_test_split
x_over_train,x_over_test,y_over_train,y_over_test = train_test_split(X,Y,test_size=0.2,random_state=3)


# In[214]:


from sklearn.metrics import mean_squared_error

def rmse(y_,y):
    return mean_squared_error(y_,y)**0.5

def rmse_scorer(model,X,Y):
    y_ = model.predict(X)
    return rmse(y_,Y)
from xgboost import XGBRegressor
from hyperopt import hp,fmin,tpe
from sklearn.model_selection import cross_val_score, KFold

def objective(params):
    params = {
        'n_estimators' : int(params['n_estimators']),
        'max_depth' : int(params['max_depth']),
        'learning_rate' : float(params['learning_rate'])
    }
    
    clf = XGBRegressor(**params,n_jobs=4)
    score = cross_val_score(clf, X, Y, scoring = rmse_scorer, cv=KFold(n_splits=3)).mean()
    print("Parmas {} - {}".format(params,score))
    return score

space = {
    'n_estimators': hp.quniform('n_estimators', 50, 1000, 50),
    'max_depth': hp.quniform('max_depth', 4, 20, 4),
    'learning_rate' : hp.uniform('learning_rate',0.05, 0.2) 
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[157]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




