
# coding: utf-8

# In[1]:


#I would be working on a currency exchange rate dataset spaning years 2016 through 2019 for 18 different countries. 
#All the numbers in this dataset are based upon the US dollar conversion rate. The aim of this analysis would be to look for
#hidden patterns amongst the four years to give an idea about how different or similar the rates were.


#The dataset included is : "exchange_rates_worldwide.csv". This has been downloaded from the Kaggle Open Datasets.


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

get_ipython().magic('matplotlib inline')


# In[3]:


#read the included data
import pandas as pd
df_rates = pd.read_csv('exchange_rates_worldwide.csv')
display(df_rates.head())


# In[4]:


#Visualize the data to get some idea at what we are looking at 

fig, axes = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
countries = df_rates.columns.difference(['Countries'])
for i in range(len(countries)):
    sns.barplot(x=countries[i], y='Countries', data=df_rates, ax=axes[i])
    axes[i].set_ylabel("")
fig.suptitle("Exchange Rates ($)")


# In[5]:


#This visualization does not give much idea about how the data is varies per year. 
#Lets try and answer this question using Principal Component Analysis


# In[6]:


#Step 1: Transform the data into mean centered data.
years = ['2016', '2017', '2018', '2019']
countries = df_rates['Countries']
X_raw = df_rates[years].values.T
print('X_raw:', X_raw.shape)


# In[8]:


#calculate the mean centered data
X = X_raw - np.mean(X_raw,axis=0)


# In[9]:


#mean centered data
X


# In[11]:


#get the SVD of the matrix 
U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
print("U:", U.shape)
print("Sigma:", Sigma.shape)
print("VT:", VT.shape)


# In[12]:


s = min(X_raw.shape)


# In[13]:


s


# In[14]:


m, d = X.shape
k_approx = 3

#plot the first 2 singular vectors
fig, axs = plt.subplots(1, k_approx, sharex=True, sharey=True,
                        figsize=(2.5*k_approx, 2.5))

for k in range(k_approx):
    v_k = VT[k, :].T
    print(f'* k = {k}:\n{v_k}\n')
    axs[k].scatter(np.arange(max(m, d)), v_k)
    axs[k].set_title(f'k = {k}')


# In[16]:


#Entries of the 1st singular vector with the largest magnitude
print("Entries of the 1st singular vector with the largest magnitude")
print(countries[[9,10,14]])

print("Entries of the 2nd singular vector with the largest magnitude")
print(countries[[8,9,14]])


# In[18]:


#Project first 2 singular vectors into 2-D
Y_k = X.dot(VT[:2,:].T)

#Print the first 2 
for x,y, label in zip(Y_k[:,0], Y_k[:,1],years):
    print(f'*{label}: ({x},{y})')


# In[20]:


#Plot the different years to see which year was different than the others
# Plot
fig = plt.figure(figsize=(3, 3))
plt.scatter(Y_k[:, 0], Y_k[:, 1])
for x, y, label in zip(Y_k[:, 0], Y_k[:, 1], years):
    plt.annotate(label, xy=(x, y))
ax = plt.gca()
ax.axis('square');


# In[ ]:


#From the above visualization we can see that the currency exchange rates for 2019 and 2016 are very closeby.
#2017 and 2018 are outliers. There can be many economic and political reasons causing this, which would need to be investigated by a subject matter expert to know more.
#But this data analysis does give a good starting point for further in-depth study.

