
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[4]:


import tensorflow as tf


# In[5]:


tf


# In[6]:


import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[13]:


# Reading data from the text file
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]


# In[15]:


# Training the model
body_reg = linear_model.LinearRegression()
# type(x_values)
body_reg.fit(x_values, y_values)


# In[18]:


# Plotting results in a graph
plt.scatter(x_values, y_values)
predicted_brain_size = body_reg.predict(x_values)
plt.plot(x_values,predicted_brain_size)
plt.show()

