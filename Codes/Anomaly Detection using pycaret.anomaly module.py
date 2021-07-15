#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
dataset=pd.read_csv("nasa.csv")


# In[23]:


dataset


# In[24]:


dataset.shape


# In[25]:


dataset.describe()


# In[26]:


data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[27]:


from pycaret.anomaly import *

exp_ano101 = setup(data, normalize = True, 
                   session_id = 123)


# # Isolation forest

# In[28]:


iforest = create_model('iforest', fraction = 0.025)


# In[29]:


print(iforest)


# In[30]:


models()


# In[31]:


result_iforest = assign_model(iforest)
result_iforest.head()


# In[51]:


plot_model(iforest, plot = 'tsne')


# In[33]:


plot_model(iforest, plot = 'umap')


# # Predictions for unseen data

# In[13]:


unseen_predictions = predict_model(iforest, data=data_unseen)
unseen_predictions


# # Predictions for training data

# In[14]:


unseen_predictions = predict_model(iforest, data=data_unseen)
unseen_predictions.head()


# # Evaluating the model

# In[19]:


evaluate_model(iforest)


# # Saving the model

# In[15]:


save_model(iforest,'Saved Isolation Forest Model')


# # Loading the model

# In[16]:


saved_iforest = load_model('Saved Isolation Forest Model')


# # New prediction of loaded model on the same unseen data

# In[17]:


new_prediction = predict_model(saved_iforest, data=data_unseen)


# In[18]:


new_prediction


# # One class SVM

# In[35]:


ocsvm = create_model('svm', fraction = 0.025)


# In[36]:


print(ocsvm)


# In[37]:


result_ocsvm = assign_model(ocsvm)
result_ocsvm


# In[38]:


evaluate_model(ocsvm)


# In[39]:


prediction = predict_model(ocsvm,data= data_unseen)


# In[40]:


prediction


# # PCA

# In[41]:


pca = create_model('pca', fraction = 0.025)


# In[42]:


print(pca)


# In[43]:


result_pca = assign_model(pca)


# In[44]:


result_pca


# In[45]:


evaluate_model(pca)


# In[46]:


prediction = predict_model(pca, data=data_unseen)


# In[47]:


prediction


# In[52]:


pycaret.anomaly.get_outliers(data=data_unseen)

