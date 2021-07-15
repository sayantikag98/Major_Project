#!/usr/bin/env python
# coding: utf-8

# # Converting .mat file to pandas dataframe 
# ## Reference: https://towardsdatascience.com/how-to-load-matlab-mat-files-in-python-1f200e1287b5 

# In[ ]:


import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# In[ ]:


data = loadmat('mill.mat')


# In[ ]:


data.keys()


# In[ ]:


type(data['mill']),data['mill'].shape


# In[ ]:


data_1 = [[row.flat[0] for row in line] for line in data['mill'][0]]
columns = ['case','run','VB','time','DOC','feed','material','smcAC','smcDC','vib_table','vib_spindle','AE_table','AE_spindle']
df_t = pd.DataFrame(data_1, columns=columns)


# # Dataset in pandas dataframe

# In[ ]:


df_t


# # Data Processing

# In[ ]:


df_t.isnull().sum()  ### checking how many NaN values are present in the entire dataframe


# In[ ]:


### filling up the NaN values using 0 or an interpolation method

df_t['VB'].fillna(0,inplace=True)

# df_t['VB'].interpolate(method='linear',inplace=True)


# In[ ]:


df_t.isnull().sum()


# In[ ]:


df_t.describe()


# In[ ]:


df_t['smcDC'].value_counts()


# In[ ]:


df_t


# # Data Visualization

# ## Using matplotlib library

# ## Plot for flank wear

# In[ ]:


## Pivot the dataframe so that the column will represent the different runs of a case and the row will represent the 
## respective values of the flank wear

df_pivot = pd.pivot_table(
	df_t,
	values="VB",
	index="case",
	columns="run"
)

# Plot a bar_plot using df_pivot 
ax = df_pivot.plot(kind="bar",width=1.5)
fig = ax.get_figure()
# Change the plot dimensions (width, height)
fig.set_size_inches(20, 10)
# Change the axes labels
ax.set_xlabel("case")
ax.set_ylabel("VB")

# Use this to show the plot in a new window
plt.show()


# ## Using seaborn library

# ## Plot for flank wear

# In[ ]:


sns.set_theme(style="dark")
ax = sns.catplot(
    data=df_t, kind="bar",
    x="case", y="VB", hue="run",
    ci="sd", palette="dark", alpha=.6, height=5, aspect=2)


# ## Plot for AC spindle motor current

# In[ ]:


ax = sns.catplot(
    data=df_t, kind="bar",
    x="case", y="smcAC", hue="run",
    ci="sd", palette="dark", alpha=.6, height=5, aspect=2)


# ## Plot for DC spindle motor current 
# 

# In[ ]:


ax = sns.catplot(
    data=df_t, kind="bar",
    x="case", y="smcDC", hue="run",
    ci="sd", palette="dark", alpha=.6, height=5, aspect=2)


# ## Plot for Table vibration

# In[ ]:


ax = sns.catplot(
    data=df_t, kind="bar",
    x="case", y="vib_table", hue="run",
    ci="sd", palette="dark", alpha=.6, height=5, aspect=2)


# ## Plot for Spindle vibration 

# In[ ]:


ax = sns.catplot(
    data=df_t, kind="bar",
    x="case", y="vib_spindle", hue="run",
    ci="sd", palette="dark", alpha=.6, height=5, aspect=2)


# ## Plot for Acoustic emission at table 

# In[ ]:


ax = sns.catplot(
    data=df_t, kind="bar",
    x="case", y="AE_table", hue="run",
    ci="sd", palette="dark", alpha=.6, height=5, aspect=2)


# ## Plot for Acoustic emission at spindle 

# In[ ]:


ax = sns.catplot(
    data=df_t, kind="bar",
    x="case", y="AE_spindle", hue="run",
    ci="sd", palette="dark", alpha=.6, height=5, aspect=2)


# In[ ]:


df_t.columns   ### getting all the column names


# # Data Cleaning

# In[ ]:


df_train = df_t[['VB','smcAC','smcDC','vib_table','vib_spindle','AE_table','AE_spindle']]


# In[ ]:


df_train


# In[ ]:


df_train.to_csv("nasa.csv", index=None)


# In[ ]:


df_train.shape[1]


# # DataFrame to numpy arrays

# In[ ]:


X = np.array(df_train)


# In[ ]:


X


# # Train test split

# In[ ]:


X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 1)


# In[ ]:


X_train = X_train.tolist()
X_test = X_test.tolist()


# In[ ]:


X_train


# # Pytorch Implementation of AutoEncoder (Stacked AutoEncoder)

# ## Tensors are arrays that contain elements of a single data type

# ## Creation of PyTorch tensors

# In[ ]:


X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)


# In[ ]:


X_train


# In[ ]:


X_test


# ## Creating the neural network architecture

# In[ ]:


class SAE (nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(df_train.shape[1], 5)  ## first layer of encoder
        self.fc2 = nn.Linear(5, 2)  ## second layer of encoder
        self.fc3 = nn.Linear(2, 5)  ## first layer of decoder
        self.fc4 = nn.Linear(5, df_train.shape[1]) ## second layer of decoder
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.fc1(x)) ## Encoding
        x = self.activation(self.fc2(x)) ## Encoding
        x = self.activation(self.fc3(x)) ## Decoding
        x = self.fc4(x)
        return x
           
    


# In[ ]:


sae = SAE() ## object of SAE (AutoEncoder) class


# In[ ]:


criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)


# In[ ]:


X_train[0]


# In[ ]:


df_train


# ## Training the SAE

# In[ ]:


epoch_num = 200
for epoch in range(1,epoch_num+1):
    train_loss = 0
    s = 0.
    for i in range(len(df_train)):
        input = Variable(X_train[i]).unsqueeze(0)
        target = input.clone()
        if(torch.sum(target.data > 0) >0):
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion (output,target)
            mean_corrector = df_train.shape[1]/float(torch.sum(target.data>0) + 1e-10)
            loss.backward()
#             train_loss += np.sqrt(loss.data[0] * mean_corrector)
#             s+=1.
#             optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
            
    
    


# ## Testing the sae

# In[ ]:


test_loss = 0
s = 0.
for i in range(len(df_train)):
    input = Variable(X_train[i]).unsqueeze(0)
    target = Variable(X_test[i])
    if(torch.sum(target.data > 0) >0):
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion (output,target)
        mean_corrector = df_train.shape[1]/float(torch.sum(target.data>0) + 1e-10)
        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        s+=1.
    
print('test_loss: '+str(test_loss/s))
    

