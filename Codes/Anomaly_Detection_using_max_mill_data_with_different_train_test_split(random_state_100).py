#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset=pd.read_csv("mill_max.csv")


# In[2]:


dataset.drop(columns={'case','run','time','DOC','feed','material'},inplace = True)


# In[3]:


dataset


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
dataset.plot(subplots=True,linewidth=3.0,figsize=(50,50), markevery=[17],marker='*',mfc='black',ms='20.0')
plt.legend(loc="upper left", fontsize='large')
plt.figure()
plt.show()


# In[6]:


dataset.iloc[17,:]


# In[7]:


dataset.boxplot(figsize=(15,15),grid=True)
plt.show()


# In[8]:


import numpy as np
pos=[]
count=0
for i in dataset.columns:
  x=dataset.loc[:,i]
  iqr=np.subtract(*np.percentile(x, [75, 25]))
  q1=np.percentile(x,25)
  q3=np.percentile(x,75)
  o1=q1-(1.5*iqr)
  o2=q3+(1.5*iqr)
  for j in range(len(dataset)):
    if ((x[j]<o1) | (x[j]>o2)):
      pos.append(True)   ## outliers would be assigned a value true
      count=count+1
    else:
      pos.append(False)
  print(f"The column {i} has {(count/len(dataset))*100}% outlier")


# In[9]:



plt.figure(figsize=(30,30))
dataset.hist(grid=True,bins=15,color='steelblue', edgecolor='black', linewidth=2.0,sharex=False)
plt.show()


# In[10]:


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 6))
corr = dataset.corr(method="pearson")
sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                linewidths=.05)
f.subplots_adjust(top=0.93)
f.suptitle('Pearson Correlation Heatmap', fontsize=14)


# In[11]:


plt.figure(figsize=(10,10))
sns.heatmap(dataset,cmap='RdYlGn_r')
plt.show()


# In[12]:


sns.distplot(dataset.VB)


# In[13]:


sns.distplot(dataset.smc_AC)


# In[14]:


sns.distplot(dataset.smc_DC)


# In[15]:


sns.distplot(dataset.vib_table)


# In[16]:


sns.distplot(dataset.vib_spindle)


# In[17]:


sns.distplot(dataset.AE_table)


# In[18]:


sns.distplot(dataset.AE_spindle)


# In[19]:


cols = ['VB', 'smc_AC','smc_DC', 'vib_table', 'vib_spindle', 'AE_table','AE_spindle']
sns.pairplot(dataset[cols], height=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))


# In[20]:


dataset.shape


# In[21]:


dataset.describe()


# In[22]:


dataset.isna().sum()


# In[23]:


dataset


# In[24]:


data = dataset.sample(frac=0.95, random_state=100) ## training
data_unseen = dataset.drop(data.index)  ## testing
print(data_unseen.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[25]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[26]:


get_ipython().system('pip install pycaret')


# In[27]:


from pycaret.anomaly import *

exp_ano101 = setup(data, normalize = True, 
                   session_id = 123)


# # Isolation forest

# In[28]:


iforest = create_model('iforest', fraction = 0.025)  ## model creation


# In[29]:


print(iforest)


# In[30]:


models()


# In[31]:


result_iforest = assign_model(iforest)  ## model training
result_iforest.head()


# In[32]:


plot_model(iforest, plot = 'tsne', label=True, feature='smc_DC')  ## plotting


# In[33]:


dataset['smc_DC']


# In[34]:


plot_model(iforest, plot = 'umap', feature = 'smc_DC')


# # Predictions for unseen data

# In[35]:


predictions = predict_model(iforest, data=data_unseen)  
predictions


# In[36]:


iforest_unseen = list(predictions.loc[predictions['Anomaly']==1].index)
iforest_unseen


# # Predictions for training data

# In[37]:


predictions = predict_model(iforest, data=data)
predictions = predictions.loc[predictions['Anomaly']==1]
predictions


# In[38]:


iforest_seen = list(predictions.index)
iforest_seen


# # Evaluating the model

# In[39]:


evaluate_model(iforest)


# # Saving the model

# In[40]:


save_model(iforest,'Saved Isolation Forest Model')


# # Loading the model

# In[41]:


saved_iforest = load_model('Saved Isolation Forest Model')


# # New prediction of loaded model on the same unseen data

# In[42]:


new_prediction = predict_model(saved_iforest, data=data_unseen)


# In[43]:


new_prediction


# # One class SVM

# In[44]:


ocsvm = create_model('svm', fraction = 0.025)


# In[45]:


print(ocsvm)


# In[46]:


result_ocsvm = assign_model(ocsvm)
result_ocsvm


# In[47]:


result_ocsvm['Anomaly_Score'].hist(bins=100)


# In[48]:


evaluate_model(ocsvm)


# In[49]:


prediction = predict_model(ocsvm,data= data_unseen) ## predictions for unseen data


# In[50]:


prediction


# In[51]:


ocsvm_unseen = list(prediction.loc[prediction['Anomaly']==1].index)
ocsvm_unseen


# In[52]:


prediction_training = predict_model(ocsvm, data) ## predictions for seen data


# In[53]:


prediction = prediction_training.loc[prediction_training['Anomaly']==1]  ## anomalous data
prediction


# In[54]:


ocsvm_seen = list(prediction.index)
ocsvm_seen


# # PCA

# In[55]:


pca = create_model('pca', fraction = 0.025)


# In[56]:


print(pca)


# In[57]:


result_pca = assign_model(pca)


# In[58]:


result_pca


# In[59]:


evaluate_model(pca)


# In[60]:


prediction = predict_model(pca, data=data_unseen) ## predictions for unseen data


# In[61]:


prediction


# In[62]:


prediction = prediction.loc[prediction['Anomaly']==1]
prediction


# In[63]:


pca_unseen = list(prediction.index)
pca_unseen


# In[64]:


prediction = predict_model(pca, data = data)  ## predictions for seen data


# In[65]:


prediction


# In[66]:


prediction = prediction.loc[prediction['Anomaly']==1]  ## Anomalous data
prediction


# In[67]:


pca_seen = list(prediction.index)
pca_seen


# #KNN

# In[68]:


knn= create_model(model='knn',fraction=0.025)


# In[69]:


print(knn)


# In[70]:


result_knn=assign_model(knn)


# In[71]:


result_knn


# In[72]:


evaluate_model(knn)


# In[73]:


prediction = predict_model(data=data_unseen,model=knn)
prediction


# In[74]:


prediction = prediction.loc[prediction['Anomaly']==1]
prediction


# In[75]:


knn_unseen = list(prediction.index)
knn_unseen


# In[76]:


prediction = predict_model(data=data,model = knn)  ## seen data predictions


# In[77]:


prediction = prediction.loc[prediction['Anomaly']==1]  ## anomalous data
prediction


# In[78]:


knn_seen = list(prediction.index) ## anomalous data index
knn_seen


# In[79]:


all_list = [iforest_seen, iforest_unseen, ocsvm_seen, ocsvm_unseen, pca_seen, pca_unseen, knn_seen, knn_unseen]


# In[80]:


df_result = pd.DataFrame(all_list).T
df_result.columns = df_result.columns.astype(str)
df_result.rename(columns = {'0':'iforest_seen', '1':'iforest_unseen', '2':'ocsvm_seen', '3':'ocsvm_unseen', '4':'pca_seen', '5':'pca_unseen','6':'knn_seen','7':'knn_unseen'}, inplace= True)


# In[81]:


df_result = df_result.astype('Int64')


# In[82]:


df_result


# In[83]:


data.iloc[25,:]


# In[84]:


data.iloc[66,:]


# In[85]:


data.iloc[87,:]


# In[86]:


data_unseen.iloc[7,:]


# #Comparing different prediction results

# In[87]:


result_iforest


# In[88]:


result_iforest['Anomaly_Score'].hist(bins=100)


# In[89]:


from pycaret.classification import *


# In[90]:


classify = setup(data=result_iforest, target='Anomaly')


# In[91]:


result_models = compare_models(sort='F1')


# In[92]:


result_models


# In[93]:


pip install shap


# In[94]:


interpret_model(result_models)


# In[ ]:




