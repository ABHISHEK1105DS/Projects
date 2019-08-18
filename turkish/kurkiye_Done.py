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
from sklearn import linear_model,datasets
from sklearn.metrics import mean_squared_error


# In[5]:


da = pd.read_csv(r"D:\ml\turkish\turkiye.csv")


# In[7]:


da


# In[8]:


da.head()


# In[10]:


def missing_percentage(data1, col_name = "Missing value (%)"):
    # Calculating the missing percentage
    missing_df1 = pd.DataFrame(data1.isnull().sum() /len(data1)*100, columns = [col_name])
    # Forming the output dataframe
    missing_df = pd.DataFrame({'Data': missing_df1.iloc[:, 0]})
    return missing_df

missing_percentage(da)


# In[11]:


"""
Exploratory Data Analysis
As we have checked our data for the missing values, let shift our focus in understanding the data in much better way. Ww will be using visualization in order to do Exploratory Data Analysis(EDA). EDA is an approach for data analysis that employs a variety of techniques mostly graphical to

Maximize insight into a data set
Uncover underlying structure
Extract important variables
Detect outliers and anomalies
Test underlying assumptions
Develop parsimonious models
Determine optimal factor settings"""


# In[15]:



da.isnull().sum()


# In[16]:


da.head()


# In[17]:


da['instr'].value_counts().plot.bar()


# In[46]:



da['class'].value_counts().plot.bar()


# In[84]:


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
        sns.distplot(da.iloc[:, i], ax=ax)

    # Show the plot
    plt.show()
    
subplot_hist(da.iloc[:, :5], row = 4, column = 3, title = "Histogram of Dataset")


# In[ ]:





# In[86]:


# for finding skew value
from scipy import stats as st 

# Computing the skewness into dataFrame
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
skewness_check(da.iloc[:, :5])


# In[ ]:





# In[ ]:





# In[116]:


da[["attendance"]] += 0.1
da[["nb.repeat"]]+=.1
def boxcox_trans(data):
    for i in range(data.shape[1]):
        data.iloc[:, i], _ = st.boxcox(data.iloc[:, i])
    return data
# Subset the predcitors
red_trans = da.copy(deep = True)
red_trans.iloc[:, :-1] = boxcox_trans(red_trans.iloc[:, :5])
skewness_check(red_trans.iloc[:, :5])


# In[ ]:





# In[ ]:





# In[101]:


print('Skewness',red_trans['nb.repeat'].skew())


# In[ ]:





# In[ ]:





# In[115]:


print('Skewness',red_trans['nb.repeat'].skew())


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





# In[67]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[117]:



red_trans['nb.repeat'].plot()
print('Skewness',red_trans['nb.repeat'].skew())


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





# In[120]:


dataset_questions = da.iloc[:,5:33]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[121]:



dataset_questions


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[114]:


#lets do a PCA for feature dimensional reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(dataset_questions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


from sklearn.cluster import KMeans
res=[]
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset_questions_pca)
    res.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.plot(range(1, 7), res,marker = "o")
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('res')
plt.show()


# In[ ]:





# In[76]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(dataset_questions_pca)


# In[ ]:





# In[77]:


# Visualising the clusters
plt.scatter(dataset_questions_pca[y_kmeans == 0, 0], dataset_questions_pca[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(dataset_questions_pca[y_kmeans == 1, 0], dataset_questions_pca[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(dataset_questions_pca[y_kmeans == 2, 0], dataset_questions_pca[y_kmeans == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'blue', label = 'Centroids')
plt.title('Clusters of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:


dataset_questions_pca.shape


# In[78]:


da.skew()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


import  collections
collections.Counter(y_kmeans)


# In[32]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(dataset_questions_pca, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('questions')
plt.ylabel('Euclidean distances')
plt.show()


# In[33]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(dataset_questions_pca)
X = dataset_questions_pca
# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.title('Clusters of STUDENTS')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()


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




