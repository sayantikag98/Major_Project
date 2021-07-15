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


# In[24]:


sns.displot(dataset.VB)


# In[27]:


sns.displot(dataset.AE_spindle)


# In[28]:


cols = ['VB', 'smc_AC','smc_DC', 'vib_table', 'vib_spindle', 'AE_table','AE_spindle']
sns.pairplot(dataset[cols], height=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))


# In[29]:


dataset.shape


# In[30]:


dataset.describe()


# In[31]:


dataset.isna().sum()


# In[34]:


dataset


# In[35]:


data = dataset.sample(frac=0.95, random_state=350) ## training
data_unseen = dataset.drop(data.index)  ## testing
print(data_unseen.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[36]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[37]:


get_ipython().system('pip install pycaret')


# In[38]:


from pycaret.anomaly import *

exp_ano101 = setup(data, normalize = True, 
                   session_id = 123)


# # Isolation forest

# In[39]:


iforest = create_model('iforest', fraction = 0.025)  ## model creation


# In[40]:


print(iforest)


# In[41]:


models()


# In[42]:


result_iforest = assign_model(iforest)  ## model training
result_iforest.head()


# In[43]:


plot_model(iforest, plot = 'tsne', label=True, feature='smc_DC')  ## plotting


# In[44]:


dataset['smc_DC']


# In[45]:


plot_model(iforest, plot = 'umap', feature = 'smc_DC')


# # Predictions for unseen data

# In[46]:


predictions = predict_model(iforest, data=data_unseen)  
predictions


# In[47]:


iforest_unseen = list(predictions.loc[predictions['Anomaly']==1].index)
iforest_unseen


# # Predictions for training data

# In[48]:


predictions = predict_model(iforest, data=data)
predictions = predictions.loc[predictions['Anomaly']==1]
predictions


# In[49]:


iforest_seen = list(predictions.index)
iforest_seen


# # Evaluating the model

# In[50]:


evaluate_model(iforest)


# # Saving the model

# In[51]:


save_model(iforest,'Saved Isolation Forest Model')


# # Loading the model

# In[52]:


saved_iforest = load_model('Saved Isolation Forest Model')


# # New prediction of loaded model on the same unseen data

# In[53]:


new_prediction = predict_model(saved_iforest, data=data_unseen)


# In[54]:


new_prediction


# # One class SVM

# In[55]:


ocsvm = create_model('svm', fraction = 0.025)


# In[56]:


print(ocsvm)


# In[57]:


result_ocsvm = assign_model(ocsvm)
result_ocsvm


# In[58]:


result_ocsvm['Anomaly_Score'].hist(bins=100)


# In[59]:


evaluate_model(ocsvm)


# In[60]:


prediction = predict_model(ocsvm,data= data_unseen) ## predictions for unseen data


# In[61]:


prediction


# In[62]:


ocsvm_unseen = list(prediction.loc[prediction['Anomaly']==1].index)
ocsvm_unseen


# In[63]:


prediction_training = predict_model(ocsvm, data) ## predictions for seen data


# In[64]:


prediction = prediction_training.loc[prediction_training['Anomaly']==1]  ## anomalous data
prediction


# In[65]:


ocsvm_seen = list(prediction.index)
ocsvm_seen


# # PCA

# In[66]:


pca = create_model('pca', fraction = 0.025)


# In[67]:


print(pca)


# In[68]:


result_pca = assign_model(pca)


# In[69]:


result_pca


# In[70]:


evaluate_model(pca)


# In[71]:


prediction = predict_model(pca, data=data_unseen) ## predictions for unseen data


# In[72]:


prediction


# In[73]:


prediction = prediction.loc[prediction['Anomaly']==1]
prediction


# In[74]:


pca_unseen = list(prediction.index)
pca_unseen


# In[75]:


prediction = predict_model(pca, data = data)  ## predictions for seen data


# In[76]:


prediction


# In[77]:


prediction = prediction.loc[prediction['Anomaly']==1]  ## Anomalous data
prediction


# In[78]:


pca_seen = list(prediction.index)
pca_seen


# #KNN

# In[79]:


knn= create_model(model='knn',fraction=0.025)


# In[80]:


print(knn)


# In[81]:


result_knn=assign_model(knn)


# In[82]:


result_knn


# In[83]:


evaluate_model(knn)


# In[84]:


prediction = predict_model(data=data_unseen,model=knn)
prediction


# In[85]:


prediction = prediction.loc[prediction['Anomaly']==1]
prediction


# In[86]:


knn_unseen = list(prediction.index)
knn_unseen


# In[87]:


prediction = predict_model(data=data,model = knn)  ## seen data predictions


# In[88]:


prediction = prediction.loc[prediction['Anomaly']==1]  ## anomalous data
prediction


# In[89]:


knn_seen = list(prediction.index) ## anomalous data index
knn_seen


# In[90]:


all_list = [iforest_seen, iforest_unseen, ocsvm_seen, ocsvm_unseen, pca_seen, pca_unseen, knn_seen, knn_unseen]


# In[91]:


df_result = pd.DataFrame(all_list).T
df_result.columns = df_result.columns.astype(str)
df_result.rename(columns = {'0':'iforest_seen', '1':'iforest_unseen', '2':'ocsvm_seen', '3':'ocsvm_unseen', '4':'pca_seen', '5':'pca_unseen','6':'knn_seen','7':'knn_unseen'}, inplace= True)


# In[92]:


df_result = df_result.astype('Int64')


# In[93]:


df_result


# In[94]:


data.iloc[25,:]


# In[95]:


data.iloc[66,:]


# In[96]:


data.iloc[87,:]


# In[97]:


data_unseen.iloc[7,:]


# #Comparing different prediction results

# In[98]:


result_iforest


# In[99]:


result_iforest['Anomaly_Score'].hist(bins=100)


# In[100]:


from pycaret.classification import *


# In[101]:


classify = setup(data=result_iforest, target='Anomaly')


# In[102]:


result_models = compare_models(sort='F1')


# In[103]:


result_models


# In[104]:


pip install shap


# In[ ]:





# In[ ]:




