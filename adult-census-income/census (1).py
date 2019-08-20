#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


data=pd.read_csv(r"D:\ml\adult-census-income\adult.csv")


# In[ ]:





# In[3]:


data.isnull().sum()


# In[4]:


# Setting all the categorical columns to type category
for col in set(data.columns) - set(data.describe().columns):
    data[col] = data[col].astype('category')
    
print('## 1.1. Columns and their types')
print(data.info())


# In[ ]:





# In[5]:


data.head()


# In[ ]:





# In[7]:


corrmat = data.corr()
fig,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


data.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


print('## 1.4. Missing values')
for i,j in zip(data.columns,(data.values.astype(str) == '?').sum(axis = 0)):
    if j > 0:
        print(str(i) + ': ' + str(j) + ' records')


# In[ ]:





# In[ ]:





# In[11]:


"""Treating Missing Values by predicting them
I fill the missing values in each of the three columns by predicting their values. For each of the three columns, I use all the attributes (including 'income') as independent variables and treat that column as the dependent variable, making it a multi-class classification task. I use three classification algorithms, namely, logistic regression, decision trees and random forest to predict the class when the value is missing (in this case a '?'). I then take a majority vote amongst the three classifiers to be the class of the missing value. In case of a tie, I pick the majority class of that column using the entire dataset."""


# In[12]:


def oneHotCatVars(df, df_cols):
    
    df_1 = adult_data = df.drop(columns = df_cols, axis = 1)
    df_2 = pd.get_dummies(df[df_cols])
    
    return (pd.concat([df_1, df_2], axis=1, join='inner'))


# In[ ]:





# In[13]:



print('## 1.5. Correlation Matrix')

display(data.corr())

print('We see that none of the columns are highly correlated.')
print('### 1.4.1. Filling in missing values for Attribute workclass')


# In[ ]:





# In[14]:


corrmat = data.corr()
fig,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


test_data = data[(data.workclass.values == '?')].copy()
test_label = test_data.workclass


# In[ ]:





# In[16]:


train_data = data[(data.workclass.values != '?')].copy()
train_label = train_data.workclass


# In[17]:


test_data.drop(columns = ['workclass'], inplace = True)


# In[18]:


train_data.drop(columns = ['workclass'], inplace = True)


# In[19]:


train_data = oneHotCatVars(train_data, train_data.select_dtypes('category').columns)


# In[20]:


test_data = oneHotCatVars(test_data, test_data.select_dtypes('category').columns)


# In[ ]:





# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


# In[22]:



test_data = data[(data.workclass.values == '?')].copy()
test_label = test_data.workclass

train_data = data[(data.workclass.values != '?')].copy()
train_label = train_data.workclass

test_data.drop(columns = ['workclass'], inplace = True)
train_data.drop(columns = ['workclass'], inplace = True)

train_data = oneHotCatVars(train_data, train_data.select_dtypes('category').columns)
test_data = oneHotCatVars(test_data, test_data.select_dtypes('category').columns)

log_reg = LogisticRegression()
log_reg.fit(train_data, train_label)
log_reg_pred = log_reg.predict(test_data)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_label)
clf_pred = clf.predict(test_data)

r_forest = RandomForestClassifier(n_estimators=10)
r_forest.fit(train_data, train_label)
r_forest_pred = r_forest.predict(test_data)

majority_class = data.workclass.value_counts().index[0]

pred_df =  pd.DataFrame({'RFor': r_forest_pred, 'DTree' : clf_pred, 'LogReg' : log_reg_pred})
overall_pred = pred_df.apply(lambda x: x.value_counts().index[0] if x.value_counts()[0] > 1 else majority_class, axis = 1)

data.loc[(data.workclass.values == '?'),'workclass'] = overall_pred.values
print(data.workclass.value_counts())
print(data.workclass.unique())


# In[23]:


overall_pred 


# In[24]:


majority_class 


# In[25]:


pred_df 


# In[26]:


print('### 1.4.2. Filling in missing values for Occupation occupation')

test_data = data[(data.occupation.values == '?')].copy()
test_label = test_data.occupation

train_data = data[(data.occupation.values != '?')].copy()
train_label = train_data.occupation

test_data.drop(columns = ['occupation'], inplace = True)
train_data.drop(columns = ['occupation'], inplace = True)

train_data = oneHotCatVars(train_data, train_data.select_dtypes('category').columns)
test_data = oneHotCatVars(test_data, test_data.select_dtypes('category').columns)

log_reg = LogisticRegression()
log_reg.fit(train_data, train_label)
log_reg_pred = log_reg.predict(test_data)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_label)
clf_pred = clf.predict(test_data)

r_forest = RandomForestClassifier(n_estimators=10)
r_forest.fit(train_data, train_label)
r_forest_pred = r_forest.predict(test_data)


majority_class = data.occupation.value_counts().index[0]

pred_df =  pd.DataFrame({'RFor': r_forest_pred, 'DTree' : clf_pred, 'LogReg' : log_reg_pred})
overall_pred = pred_df.apply(lambda x: x.value_counts().index[0] if x.value_counts()[0] > 1 else majority_class, axis = 1)

data.loc[(data.occupation.values == '?'),'occupation'] = overall_pred.values
print(data.occupation.value_counts())
print(data.occupation.unique())


# In[27]:


print('### 1.4.3. Filling in missing values for Native Country')

test_data = data[(data['native.country'].values == '?')].copy()
test_label = test_data['native.country']

train_data = data[(data['native.country'].values != '?')].copy()
train_label = train_data['native.country']

test_data.drop(columns = ['native.country'], inplace = True)
train_data.drop(columns = ['native.country'], inplace = True)

train_data = oneHotCatVars(train_data, train_data.select_dtypes('category').columns)
test_data = oneHotCatVars(test_data, test_data.select_dtypes('category').columns)

log_reg = LogisticRegression()
log_reg.fit(train_data, train_label)
log_reg_pred = log_reg.predict(test_data)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_label)
clf_pred = clf.predict(test_data)

r_forest = RandomForestClassifier(n_estimators=10)
r_forest.fit(train_data, train_label)
r_forest_pred = r_forest.predict(test_data)


majority_class = data['native.country'].value_counts().index[0]

pred_df =  pd.DataFrame({'RFor': r_forest_pred, 'DTree' : clf_pred, 'LogReg' : log_reg_pred})
overall_pred = pred_df.apply(lambda x: x.value_counts().index[0] if x.value_counts()[0] > 1 else majority_class, axis = 1)

data.loc[(data['native.country'].values == '?'),'native.country'] = overall_pred.values
print(data['native.country'].value_counts())
print(data['native.country'].unique())


# In[28]:


data.head()


# In[29]:


print('## 1.5. Correlation Matrix')

display(data.corr())


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


# Resetting the categories

data['workclass'] = data['workclass'].cat.remove_categories('?')
data['occupation'] = data['occupation'].cat.remove_categories('?')
data['native.country'] = data['native.country'].cat.remove_categories('?')


# In[ ]:





# In[31]:


data.head()


# In[32]:


print('## 1.5. Correlation Matrix')

display(data.corr())


# In[33]:


city_category = np.unique(data['education.num'])
city_category


# In[34]:


city_category = np.unique(data['education'])
city_category


# In[35]:


# Creating a dictionary that contain the education and it's corresponding education level
edu_level = {}
for x,y in data[['education.num','education']].drop_duplicates().itertuples(index=False):
    edu_level[y] = x


# In[36]:


print('## 2.1. Education vs Income')
education = round(pd.crosstab(data.education, data.income).div(pd.crosstab(data.education, data.income).apply(sum,1),0),2)
education = education.reindex(sorted(edu_level, key=edu_level.get, reverse=False))

ax = education.plot(kind ='bar', title = 'Proportion distribution across education levels', figsize = (10,8))
ax.set_xlabel('Education level')
ax.set_ylabel('Proportion of population')


# In[37]:


print('As the education increase income also increased')


# In[38]:


data.head()


# In[39]:


print('## 2.2 Sex vs Income')

gender = round(pd.crosstab(data.sex, data.income).div(pd.crosstab(data.sex, data.income).apply(sum,1),0),2)
gender.sort_values(by = '>50K', inplace = True)
ax = gender.plot(kind ='bar', title = 'Proportion distribution across gender levels')
ax.set_xlabel('Gender level')
ax.set_ylabel('Proportion of population')


# In[40]:


print("from this we can infere that the there is wage gap between male and female but we dont know any fixed value ")


# In[41]:


gender_workclass = round(pd.crosstab(data.workclass, [data.income, data.sex]).div(pd.crosstab(data.workclass, [data.income, data.sex]).apply(sum,1),0),2)
gender_workclass[[('>50K','Male'), ('>50K','Female')]].plot(kind = 'bar', title = 'Proportion distribution across gender for each workclass', figsize = (10,8), rot = 30)
ax.set_xlabel('Gender level')
ax.set_ylabel('Proportion of population')


# In[42]:


print("varius distribution of money on different platform where money is greater than 50k")


# In[43]:


gender_workclass = round(pd.crosstab(data.workclass, [data.income, data.sex]).div(pd.crosstab(data.workclass, [data.income, data.sex]).apply(sum,1),0),2)
gender_workclass[[('<=50K','Male'), ('<=50K','Female')]].plot(kind = 'bar', title = 'Proportion distribution across gender for each workclass', figsize = (10,8), rot = 30)
ax.set_xlabel('Gender level')
ax.set_ylabel('Proportion of population')
print("varius distribution of money on different platform where money is less than 50k")


# In[44]:


print(' 2.3. Occupation vs Income')

occupation = round(pd.crosstab(data.occupation, data.income).div(pd.crosstab(data.occupation, data.income).apply(sum,1),0),2)
occupation.sort_values(by = '>50K', inplace = True)
ax = occupation.plot(kind ='bar', title = 'Proportion distribution across Occupation levels', figsize = (10,8))
ax.set_xlabel('Occupation level')
ax.set_ylabel('Proportion of population')
print("Occupation category having ='private-house-servant 'has higher percentage of people more than 50 k")
print("Occupation category having ='exec-managerical 'has higher percentage of people more than 50 k")


# In[45]:


print(' 2.4. Workclass vs Income')

workclass = round(pd.crosstab(data.workclass, data.income).div(pd.crosstab(data.workclass, data.income).apply(sum,1),0),2)
workclass.sort_values(by = '>50K', inplace = True)
ax = workclass.plot(kind ='bar', title = 'Proportion distribution across workclass levels', figsize = (10,8))
ax.set_xlabel('Workclass level')
ax.set_ylabel('Proportion of population')


# In[46]:


print('## 2.5. Race vs Income')

race = round(pd.crosstab(data.race, data.income).div(pd.crosstab(data.race, data.income).apply(sum,1),0),2)
race.sort_values(by = '>50K', inplace = True)
ax = race.plot(kind ='bar', title = 'Proportion distribution across race levels', figsize = (10,8))
ax.set_xlabel('Race level')
ax.set_ylabel('Proportion of population')

print()


# In[47]:


print('## 2.6. Native Country')

native_country = round(pd.crosstab(data['native.country'], data.income).div(pd.crosstab(data['native.country'], data.income).apply(sum,1),0),2)
native_country.sort_values(by = '>50K', inplace = True)
ax = native_country.plot(kind ='bar', title = 'Proportion distribution across Native Country levels', figsize = (20,12))
ax.set_xlabel('Native country')
ax.set_ylabel('Proportion of population')
print("From the graph, we notice a trend in positioning of the country. South American country are at the left end of the plot, with low proportion of population that make more than 50k a year. The United States is located somewhat centrally, and at the right are countries from Europe and Asia, with higher proportion of population that make more than 50k a year.")


# In[48]:


corrmat = data.corr()
fig,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[49]:


data.head()


# In[50]:


for col in set(data.columns) - set(data.describe().columns):
    data[col] = data[col].astype('category')
    
print('## 1.1. Columns and their types')
print(data.info())


# In[59]:


data.drop(columns = ['education','fnlwgt','hours.per.week'], inplace = True)

print('* For education level, we have 2 features that convey the same meaning, \'education\'         and \'educational-num\'. To avoid the effect of this attribute on the models to be         overstated, I am not going to use the categorical education attribute.')
print('* I use the categorical Hours work column and drop the \'hour-per-week\' column')
print( '*Also, I chose not to use the \'Fnlwgt\' attribute that is used by the census,         as the inverse of sampling fraction adjusted for non-response and over or under sampling         of particular groups. This attribute does not convey individual related meaning.')


# In[60]:


print('## Box plot')
data.select_dtypes(exclude = 'category').plot(kind = 'box', figsize = (10,8))


# In[65]:


print('Normalization happens on the training dataset, by removing the mean and  scaling to unit variance. These values are stored and then later applied  to the test data before the test data is passed to the model for prediction. ')


# In[ ]:


"""
4. Model Development & Classification
4.1. Data Preparation'
One-hot encoding is the process of representing multi-class categorical features as binary features, one for each class. Although this process increases the dimensionality of the dataset, classification algorithms tend to work better on this format of data.

I use one-hot encoding to represent all the categorical features in the dataset."""


# In[66]:


# Data Prep
adult_data = data.drop(columns = ['income'])
adult_label = data.income


adult_cat_1hot = pd.get_dummies(data.select_dtypes('category'))
adult_non_cat = data.select_dtypes(exclude = 'category')

adult_data_1hot = pd.concat([adult_non_cat, adult_cat_1hot], axis=1, join='inner')


# In[69]:


#  Train - Test split
from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(adult_data_1hot, adult_label, test_size  = 0.25)


# In[70]:


# Normalization
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  

# Fitting only on training data
scaler.fit(train_data)  
train_data = scaler.transform(train_data)  

# Applying same transformation to test data
test_data = scaler.transform(test_data)


# In[71]:


def model_eval(actual, pred):
    
    confusion = pd.crosstab(actual, pred, rownames=['Actual'], colnames=['Predicted'])
    TP = confusion.loc['>50K','>50K']
    TN = confusion.loc['<=50K','<=50K']
    FP = confusion.loc['<=50K','>50K']
    FN = confusion.loc['>50K','<=50K']

    accuracy = ((TP+TN))/(TP+FN+FP+TN)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    f_measure = (2*recall*precision)/(recall+precision)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    error_rate = 1 - accuracy
    
    out = {}
    out['accuracy'] =  accuracy
    out['precision'] = precision
    out['recall'] = recall
    out['f_measure'] = f_measure
    out['sensitivity'] = sensitivity
    out['specificity'] = specificity
    out['error_rate'] = error_rate
    
    return out


# In[72]:


"""4.2. Model Development
4.2.1. Decision Tree
For the decision tree classifier, I experimented with the splitting criteria, minimum samples required to split, max depth of the tree, minimum samples required at the leaf level and the maximum features to consider when looking for the best split. The following values of the parameters attained the best accuracy during classification. Results in the table below.

Splitting criteria: Gini Index (Using Gini Index marginally outperformed Entropy with a higher accuracy.)
Min samples required to split: 5% (Best amongst 1%, 10% and 5%.)
Max Depth: None
Min samples required at leaf: 0.1 % (Best amongst 1%, 5% and 0.1%.)
Max features: number of features (Performs better than 'auto', 'log2' and 'sqrt'.)
"""


# In[73]:


print('### 3.1.1. Model Development ')

# Gini 
clf_gini = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_split = 0.05, min_samples_leaf = 0.001, max_features = None)
clf_gini = clf_gini.fit(train_data, train_label)
clf_gini_pred = clf_gini.predict(test_data)
DTree_Gini = model_eval(test_label, clf_gini_pred)
print('Desicion Tree using Gini Index : %.2f percent.' % (round(DTree_Gini['accuracy']*100,2)))


# Entropy
clf_entropy = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 0.05, min_samples_leaf = 0.001)
clf_entropy = clf_entropy.fit(train_data, train_label)
clf_entropy_pred = clf_entropy.predict(test_data)
DTree_Entropy = model_eval(test_label, clf_entropy_pred)
print('Desicion Tree using Entropy : %.2f percent.' % (round(DTree_Entropy['accuracy']*100,2)))

print('### 3.1.2. Model Evaulation ')
ovl_dtree = round(pd.DataFrame([DTree_Entropy, DTree_Gini], index = ['DTree_Entropy','DTree_Gini']),4)
display(ovl_dtree)


# In[ ]:


"""For the ANN classifier, I experimented with the activation function, the solver for weight optimization, regularization term and learning schedule for weight updates. The following values of the parameters attained the best accuracy during classification. Other parameters were neither applicable to the 'adam' solver nor did it improve the performance of the model. Results in the table below.

Activation: Logistic (Marginally outperformed 'relu', 'tanh' and 'identity' functions.)
Solver: Adam (Works well on relatively large datasets with thousands of training samples or more)
Alpha: 1e-4 (Best amongst 1, 1e-1, 1e-2, 1e-3, 1e-4 and 1e-5)
Learning Rate: 'invscaling' (Gradually decreases the learning rate at each time step 't' using an inverse scaling exponent of 'power_t'.)"""


# In[81]:


from sklearn.neural_network import MLPClassifier
ann_tanh = MLPClassifier(activation = 'tanh', solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(10, 2), random_state=1, warm_start=True)
ann_tanh.fit(train_data, train_label)                         
ann_tanh_pred = ann_tanh.predict(test_data)
ANN_TanH = model_eval(test_label, ann_tanh_pred)
print('ANN using TanH and lbfgs solver : %.2f percent.' % (round(ANN_TanH['accuracy']*100,2)))


# Relu
ann_relu = MLPClassifier(activation = 'relu', solver='adam', alpha=1e-1, 
                    hidden_layer_sizes=(5, 2), random_state=1,
                    learning_rate  = 'invscaling',
                    warm_start = True)
ann_relu.fit(train_data, train_label)                         
ann_relu_pred = ann_relu.predict(test_data)
ANN_relu = model_eval(test_label, ann_relu_pred)
print('ANN using relu and adam solver : %.2f percent.' % (round(ANN_relu['accuracy']*100,2)))

# Log
ann_log = MLPClassifier(activation = 'logistic', solver='adam', 
                    alpha=1e-4, hidden_layer_sizes=(5, 2),
                    learning_rate  = 'invscaling', 
                    random_state=1, warm_start = True)
ann_log.fit(train_data, train_label)                         
ann_log_pred = ann_log.predict(test_data)
ANN_log = model_eval(test_label, ann_log_pred)
print('ANN using logistic and adam solver : %.2f percent.' % (round(ANN_log['accuracy']*100,2)))

# Identity
ann_identity = MLPClassifier(activation = 'identity', solver='adam', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=1, warm_start = True)
ann_identity.fit(train_data, train_label)                         
ann_identity_pred = ann_identity.predict(test_data)
ANN_identity = model_eval(test_label, ann_identity_pred)
print('ANN using identity and adam solver : %.2f percent.' % (round(ANN_identity['accuracy']*100,2)))

#printmd('### 3.2.2. Model Evaulation ')
ovl_ann = round(pd.DataFrame([ANN_TanH, ANN_relu, ANN_log, ANN_identity], index = ['ANN_TanH','ANN_relu', 'ANN_log', 'ANN_identity']),4)
display(ovl_ann)


# In[ ]:


"""4.2.3. Support Vector Machine
For the SVM classifier, I experimented with the various available kernels, the penalty of the error term and the tolerance for stopping criteria. The following values of the parameters attained the best accuracy during classification. Results in the table below.

Kernel: rbf (Marginally outperformed 'linear, 'poly' and 'sigmoid' kernels.)
C, penalty of the error term: 1 (Best amongst 0.1, 0.5, 1 and 10)
Tolerance for stopping criteria: 1e-3 (Best amongst 1e-1, 1e-2, 1e-3, 1e-4 and 1e-5)"""


# In[83]:


from sklearn import svm
# rbf kernal
svm_clf_rbf = svm.SVC(kernel = 'rbf', C = 1, tol = 1e-3)
svm_clf_rbf.fit(train_data, train_label)
svm_clf_rbf_pred = svm_clf_rbf.predict(test_data)
SVM_rbf = model_eval(test_label, svm_clf_rbf_pred)
print('SVM using rbf kernel : %.2f percent.' % (round(SVM_rbf['accuracy']*100,2)))

# Linear kernel
svm_clf_linear = svm.SVC(kernel = 'linear')
svm_clf_linear.fit(train_data, train_label)
svm_clf_linear_pred = svm_clf_linear.predict(test_data)
SVM_linear = model_eval(test_label, svm_clf_linear_pred)
print('SVM using linear kernel : %.2f percent.' % (round(SVM_linear['accuracy']*100,2)))


# Poly kernal
svm_clf_poly = svm.SVC(kernel = 'poly')
svm_clf_poly.fit(train_data, train_label)
svm_clf_poly_pred = svm_clf_poly.predict(test_data)
SVM_poly = model_eval(test_label, svm_clf_poly_pred)
print('SVM using poly kernel : %.2f percent.' % (round(SVM_poly['accuracy']*100,2)))


svm_clf_sigmoid = svm.SVC(kernel = 'sigmoid')
svm_clf_sigmoid.fit(train_data, train_label)
svm_clf_sigmoid_pred = svm_clf_sigmoid.predict(test_data)
SVM_sigmoid = model_eval(test_label, svm_clf_sigmoid_pred)
print('SVM using sigmoid kernel : %.2f percent.' % (round(SVM_sigmoid['accuracy']*100,2)))



#printmd('### 3.3.2. Model Evaulation ')
ovl_svm = round(pd.DataFrame([SVM_rbf, SVM_linear, SVM_poly, SVM_sigmoid], index = ['SVM_rbf','SVM_linear', 'SVM_poly', 'SVM_sigmoid']),4)
display(ovl_svm)


# In[ ]:


"""
4.2.4. Ensemble Models
4.2.4.1. Random Forest
For the random forests classifier, I experimented with the number of trees, splitting criteria, minimum samples required to split, max depth of the tree, minimum samples required at the leaf level and the maximum features to consider when looking for the best split. The following values of the parameters attained the best accuracy during classification. Results in the table below.

Num estimators: 100 (Best amongst 10, 50 and 100)
Splitting criteria: Gini Index (Using Gini Index marginally outperformed Entropy with a higher accuracy.)
Min samples required to split: 5% (Best amongst 1%, 10% and 5%.)
Max Depth: None
Min samples required at leaf: 0.1 % (Best amongst 1%, 5% and 0.1%.)
Max features: number of features (Performs better than 'auto', 'log2' and 'sqrt'.)"""


# In[84]:


# Gini
r_forest_gini = RandomForestClassifier(n_estimators=100, criterion = 'gini', max_features = None,  min_samples_split = 0.05, min_samples_leaf = 0.001)
r_forest_gini.fit(train_data, train_label)
r_forest_gini_pred = r_forest_gini.predict(test_data)
rforest_gini = model_eval(test_label, r_forest_gini_pred)
print('Random Forest using Gini Index : %.2f percent.' % (round(rforest_gini['accuracy']*100,2)))

# Entropy
r_forest_entropy = RandomForestClassifier(n_estimators=100, criterion = 'entropy', max_features = None,  min_samples_split = 0.05, min_samples_leaf = 0.001)
r_forest_entropy.fit(train_data, train_label)
r_forest_entropy_pred = r_forest_entropy.predict(test_data)
rforest_entropy = model_eval(test_label, r_forest_entropy_pred)
print('Random Forest using Entropy : %.2f percent.' % (round(rforest_entropy['accuracy']*100,2)))

#printmd('### 3.4.1.2. Model Evaulation ')
ovl_rf = round(pd.DataFrame([rforest_gini, rforest_entropy], index = ['rforest_gini','rforest_entropy']),4)
display(ovl_rf)


# In[ ]:


"""4.2.4.2. Adaboost
For the adaboost classifier, I experimented with base estimator from which the boosted ensemble is built and number of estimators. The following values of the parameters attained the best accuracy during classification. Results in the table below.

Base Estimator: DecisionTreeClassifier

Num estimators: 100 (Best amongst 10, 50 and 100.)"""


# In[85]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=100)                     
ada.fit(train_data, train_label)
ada_pred = ada.predict(test_data)
adaboost = model_eval(test_label, ada_pred)
print('Adaboost : %.2f percent.' % (round(adaboost['accuracy']*100,2)))

#printmd('### 3.4.2.2. Model Evaulation ')
ovl_ada = round(pd.DataFrame([adaboost], index = ['adaboost']),4)
display(ovl_ada)


# In[86]:


"""4.2.5. Logistic Regression"""
log_reg = LogisticRegression(penalty = 'l2', dual = False, tol = 1e-4, fit_intercept = True, 
                            solver = 'liblinear')
log_reg.fit(train_data, train_label)
log_reg_pred = log_reg.predict(test_data)
logistic_reg = model_eval(test_label, log_reg_pred)
print('Logistic Regression : %.2f percent.' % (round(logistic_reg['accuracy']*100,3)))

#printmd('### 3.5.2. Model Evaulation ')
ovl_logreg = round(pd.DataFrame([logistic_reg], index = ['logistic_reg']),4)
display(ovl_logreg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# roc curve


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Overall Performance Statistics


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




