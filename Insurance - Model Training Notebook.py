#!/usr/bin/env python
# coding: utf-8

# In[5]:


from pycaret.datasets import get_data
data = get_data('insurance')


# In[4]:


get_ipython().system('pip install numpy')


# # Experiment 1

# In[2]:


from pycaret.regression import *


# In[5]:


s = setup(data, target = 'charges', session_id = 123)


# In[6]:


lr = create_model('lr')


# In[7]:


plot_model(lr)


# # Experiment 2

# In[8]:


s2 = setup(data, target = 'charges', session_id = 123,
           normalize = True,
           polynomial_features = True, trigonometry_features = True, feature_interaction=True, 
           bin_numeric_features= ['age', 'bmi'])


# In[9]:


s2[0].columns


# In[10]:


lr = create_model('lr')


# In[11]:


plot_model(lr)


# In[12]:


save_model(lr, 'deployment_28042020')


# In[13]:


deployment_28042020 = load_model('deployment_28042020')


# In[14]:


deployment_28042020


# In[16]:


import requests
url = 'https://pycaret-insurance.herokuapp.com/predict_api'
pred = requests.post(url,json={'age':55, 'sex':'male', 'bmi':59, 'children':1, 'smoker':'male', 'region':'northwest'})
print(pred.json())

