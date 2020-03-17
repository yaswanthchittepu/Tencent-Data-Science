#!/usr/bin/env python
# coding: utf-8

# ## Country wise Clustering Model

# ### Import the Requisite modules

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# ### Feature Engineering

# Load the given Data into a Pandas dataframe

# In[42]:


df_orig = pd.read_csv('topline_metrics.csv')
del df_orig['Date.1']


# Create a Copy of the Dataframe for making predictions. Also the data contains numerous entries for the same country, Date, and Platform i.e. duplicates, which are also removed in this step

# In[43]:


df = df_orig.copy()
df.drop_duplicates(subset = ["Date",'Platform','Country'],keep=False, inplace = True)


# In[44]:


chinese_map = {}
global_cnt = 0


# In[45]:


def trim(s):
    idx = s.find('(')
    if(idx >= 0):
        ns = s[0:idx]
        ns = ns.strip()
        if(ns == 'Côte dIvoire'):
            return 'Cote dIvoire'
        return ns
    else:
        if(s in chinese_map.keys()):
            return chinese_map[s]
        else:
            ns = 'chinese_name_'+str(trim.counter)
            chinese_map[s] = ns
            trim.counter += 1
            return ns
trim.counter = 0


# Creating other useful features from the given date information. Also it was observed that the Time spent field had negative entries, which doesnt make sense. Hence, these negative entries were made zero, to ensure consistency.

# In[46]:


df['Country'] = df['Country'].apply(lambda x: trim(x))
df['Date'] =  pd.to_datetime(df['Date'])
df['dayofmonth'] = df.Date.dt.day
df['dayofyear'] = df.Date.dt.dayofyear
df['dayofweek'] = df.Date.dt.dayofweek
df['month'] = df.Date.dt.month
df['year'] = df.Date.dt.year
df['weekofyear'] = df.Date.dt.weekofyear
df['is_month_start'] = (df.Date.dt.is_month_start).astype(int)
df['is_month_end'] = (df.Date.dt.is_month_end).astype(int)
df['weekend'] = ((df.Date.dt.dayofweek) // 5 == 1).astype(int)
df.loc[df['Time Spend Per Day(seconds)'] < 0,'Time Spend Per Day(seconds)'] = 0


# We need to convert the Dataframe, that is at a Country, Product, and Date level, into a Country level dataframe, so that we can segment the countries. We average the initially provided features across Country and Platform, and across Country, Platform, and Weekend, to take a pivot, to get these features at the Country level.

# In[47]:


df_new = pd.DataFrame({'Country': list(set(df.Country))})


# In[48]:


mean_metrics = df[['TRU','DAU','Items','Trans','Conversion','Cash Flow','Return Customer','Time Spend Per Day(seconds)']].groupby([df['Country'],df['Platform']]).median().reset_index()
df_new = pd.merge(df_new,mean_metrics, on='Country', how='inner')


# In[49]:


temp = df[['TRU','DAU','Items','Trans','Conversion','Cash Flow','Return Customer','Time Spend Per Day(seconds)']].groupby([df['Country'],df['Platform'],df['weekend']]).mean().reset_index()


# In[50]:


pvt = pd.pivot_table(temp, values = ['TRU','DAU','Items','Trans','Conversion','Cash Flow','Return Customer', 'Time Spend Per Day(seconds)'], index=['Country','Platform'], columns = 'weekend').reset_index()
cols = ['Country','Platform','TRU_0','TRU_1','DAU_0','DAU_1','Items_0','Items_1','Trans_0','Trans_1','Conversion_0',
        'Conversion_1','Cash Flow_0','Cash Flow_1','Return Customer_0','Return Customer_1','Time Spend Per Day(seconds)_0',
        'Time Spend Per Day(seconds)_1']
pvt.columns = pvt.columns.droplevel(0) 
pvt.columns.name = None
pvt = pvt.reset_index()  
del pvt['index']
pvt.columns = cols
pvt = pvt[['Country','Platform','TRU_1','DAU_1','Items_1','Trans_1','Conversion_1','Cash Flow_1','Return Customer_1',
        'Time Spend Per Day(seconds)_1']]
df_new = pd.merge(df_new,pvt, on=['Country','Platform'], how='inner')


# In[51]:


df_new.head()


# In[52]:


final_cols = []
c = list(df_new.columns)
final_cols = [c[0]] + [i+'_'+j for i in c[2:] for j in ['ALL','Android','IOS']]


# In[53]:


clust_pvt = pd.pivot_table(df_new, values = c[2:], index=['Country'], columns = 'Platform').reset_index()
clust_pvt.columns = clust_pvt.columns.droplevel(0)
clust_pvt.columns.name = None
clust_pvt = clust_pvt.reset_index()
del clust_pvt['index']
clust_pvt.columns = final_cols
clust_pvt.fillna(0, inplace=True)


# clust_pvt is our final Country level, dataset, on which we will be Clustering

# In[54]:


clust_pvt.head()


# ### Building a K Means Clustering Model 

# In[55]:


X = clust_pvt[list(clust_pvt.columns)[1:]].values


# Standardizing the variables, to make the model scale invariant

# In[61]:


from sklearn.preprocessing import StandardScaler
x_calls = clust_pvt.columns[1:]
scaller = StandardScaler()
matrix = pd.DataFrame(scaller.fit_transform(clust_pvt[x_calls]),columns=x_calls)
matrix['Country'] = clust_pvt['Country']


# The Number of Clusters in the K-Means Algorithm is a Hyperparameter. We evaluate the Bayesian Information Criteion (BIC), across K-Means models with different values of K, and choose the model with the lowest BIC. BIC reduced Overfitting, by penalizing the model for using a greater number of parameters.

# In[62]:


def plot_BIC(matrix,x_calls,K):
    from sklearn import mixture
    BIC=[]
    for k in K:
        model=mixture.GaussianMixture(n_components=k,init_params='kmeans')
        model.fit(matrix[x_calls])
        BIC.append(model.bic(matrix[x_calls]))
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(K,BIC,'-cx')
    plt.ylabel("BIC score")
    plt.xlabel("k")
    plt.title("BIC scoring for K-means cell's behaviour")
    return(BIC)


# In[63]:


K = range(2,31)
BIC = plot_BIC(matrix,x_calls,K)


# We see that a value of 5 for the number of clusters, has the lowest BIC

# In[19]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
cluster = KMeans(n_clusters=5,random_state=217)
matrix['cluster'] = cluster.fit_predict(matrix[x_calls])
print(matrix.cluster.value_counts())


# In[20]:


d=pd.DataFrame(matrix.cluster.value_counts())
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(d.index,d['cluster'],align='center',alpha=0.5)
plt.xlabel('Clusters')
plt.ylabel('Number of Countries')
plt.show()


# In[21]:


from sklearn.metrics.pairwise import euclidean_distances
distance = euclidean_distances(cluster.cluster_centers_, cluster.cluster_centers_)
print(distance)


# In[22]:


pca = PCA(n_components=3)
matrix['x'] = pca.fit_transform(matrix[x_calls])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_calls])[:,1]
matrix['z'] = pca.fit_transform(matrix[x_calls])[:,2]

cluster_centers = pca.transform(cluster.cluster_centers_)
cluster_centers = pd.DataFrame(cluster_centers, columns=['x', 'y', 'z'])
cluster_centers['cluster'] = range(0, len(cluster_centers))
print(cluster_centers)


# In[23]:


fig, ax = plt.subplots(figsize=(8, 6))
scatter=ax.scatter(matrix['x'],matrix['y'],c=matrix['cluster'],s=21,cmap=plt.cm.Set1_r)
ax.scatter(cluster_centers['x'],cluster_centers['y'],s=70,c='blue',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)
plt.title('Data Segmentation')
plt.show()


# In[24]:


matrix.head()


# Aggregating the features at the Cluster level, in order to analyze their characteristics

# In[25]:


cluster_df = pd.merge(df,matrix[['Country','cluster']],on='Country', how='inner')


# In[26]:


cluster_df[['TRU','DAU','Items','Trans','Conversion','Cash Flow','Return Customer','Time Spend Per Day(seconds)']].groupby([cluster_df['cluster'],cluster_df['Platform']]).median().reset_index()


# ### Insights into the Segmentation Scheme:
# 
# 1) The Countries belonging to Cluster 0, spend less amount on in-game transactions, but spend a lot of time in-game. Geographically, this makes sense, since these are small and comparitively poor countries, in Asia, Africa, and Central Europe. (Eg: Afghanistan, Somalia, Syria, Armenia)
# 
# 2) The Countries belonging to Cluster 1, contribute significantly, to the number of active users, but also spend a lot more on in-game transactions, while spending lesser time in game than Countries in cluster 0. Geographically, these countries are relatively rich and developed, and hence the insight makes sense. (Eg: United States, Japan, Malaysia, Turkey)
# 
# 3) The Countries belonging to Cluster 2, do not spend much time and money in-game. Also, these countries had some Mandarin names which makes reasoning about them geographically, quite difficult. These Countries could very well be outliers.
# 
# 4) The Countries in Cluster 3, are similar to countries in Cluster 1, in that they geographically correspond to developed nations in Europe, Americas, and Australia. These countries, do not spend much time or money in-game, unlike Cluster 1. But they spend a similar amount of money in-game to Cluster 0, while spending wasy less time in-game. These Countries havent been tapped into as much, and are excellent candidates for targeted promotions. Examples of such countries in CLuster 3, are China, Australia, Spain, Sweden etc.
# 
# 5) The Country, that belongs to Cluster 4 is India. This Country, contributes a significant proportion to the Daily Active User base, and also spends the most amount of money in-game, while spending lesser time in-game than Cluster 0. India, seems to be a valuable contributor to the business, and also has an extremely high conversion rate. The Customers in India, are fully tapped into the game, and need to be retained. This Cluster could also be grouped along with Cluster 1, keeping in mind, that Cluster 4 contains only a single country (India).

# In[28]:


# Change the Cluster key to look at the Countries assigned to that Cluster

matrix[matrix['cluster'] == 0]['Country']

