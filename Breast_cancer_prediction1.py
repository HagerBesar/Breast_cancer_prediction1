#!/usr/bin/env python
# coding: utf-8

# In[ ]:


####################################<<<<Breast_cancer_prediction>>>>>>####################################


# In[ ]:


#part(1)--By:Manar Moeanse


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


DB = pd.read_csv('Breast_cancer_data.csv')
DB


# In[3]:


DB.head(5)


# In[4]:


DB.describe()


# In[ ]:


DB.info()


# In[ ]:


#part(2)--By:Mariam Mamdoh


# In[5]:


uneff = DB[DB.diagnosis == 0]
eff = DB[DB.diagnosis == 1]
len(uneff)


# In[ ]:


len(eff)


# In[6]:


uneffected = (len(uneff)/len(DB)) *100
print('people are uneffected = ', uneffected , '% .')
effected = (len(eff)/len(DB)) *100
print('people are effected = ', effected , '% .')


# In[ ]:


#part(3)--By:Hemat Shawky.


# In[7]:


plt.scatter(DB['diagnosis'],DB['mean_area'])


# In[8]:


plt.scatter(DB['mean_area'],DB['mean_texture'])


# In[9]:


plt.scatter(DB['mean_radius'],DB['mean_perimeter'])


# In[10]:


import seaborn as sns
sns.pairplot(data=DB)


# In[ ]:


#part(4)--By:Hager Mohamed.


# In[11]:


x = DB.drop('diagnosis', 1)
y = DB['diagnosis']
x


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


# In[13]:


from sklearn.linear_model import LinearRegression
model = LinearRegression ()


# In[14]:


model.fit(x_train,y_train)


# In[15]:


pred =model.predict(x_test)


# In[16]:


from sklearn.metrics import mean_squared_error


# In[17]:


error=np.sqrt(mean_squared_error(y_pred=pred,y_true=y_test))
print(error)


# In[18]:


print(model.score(x_test,y_test))


# In[ ]:


"""" BY:
        1-Manar Moeanse.
        2-Mariam Mamdoh.
        3-Hemat Shawky.
        4-Hager Mohamed.

