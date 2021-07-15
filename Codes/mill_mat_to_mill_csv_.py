#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


annots = loadmat('drive/MyDrive/mill.mat')


# In[4]:


def array_concatenation(ind):
    new_array = np.concatenate((annots['mill'][0][0][ind],annots['mill'][0][1][ind]))
    for i in range(2,167):
        new_array = np.concatenate((new_array,annots['mill'][0][i][ind]))
    return new_array


# In[5]:


case_list = [item for sublist in array_concatenation(0).tolist() for item in sublist]
run_list = [item for sublist in array_concatenation(1).tolist() for item in sublist]
VB_list = [item for sublist in array_concatenation(2).tolist() for item in sublist]
time_list = [item for sublist in array_concatenation(3).tolist() for item in sublist]
DOC_list = [item for sublist in array_concatenation(4).tolist() for item in sublist]
feed_list = [item for sublist in array_concatenation(5).tolist() for item in sublist]
material_list = [item for sublist in array_concatenation(6).tolist() for item in sublist]


# In[6]:


smc_AC_list = [item for sublist in array_concatenation(7).tolist() for item in sublist]
smc_DC_list = [item for sublist in array_concatenation(8).tolist() for item in sublist]
vib_table_list = [item for sublist in array_concatenation(9).tolist() for item in sublist]
vib_spindle_list = [item for sublist in array_concatenation(10).tolist() for item in sublist]
AE_table_list = [item for sublist in array_concatenation(11).tolist() for item in sublist]
AE_spindle_list = [item for sublist in array_concatenation(12).tolist() for item in sublist]



# In[7]:


for i in range(167):
    if(annots['mill'][0][i][7].shape[0]==15360):
        print(i)
        print(annots['mill'][0][i][7].shape)
    else:
      break


# In[8]:


annots['mill'][0][94][7].shape


# In[9]:


annots['mill'][0][94][8].shape


# In[10]:


def value_copy(l):
  li = []
  for i,val in enumerate(l):
    if (i==94): x = 15360
    else: x = 9000
    for j in range(x):
      li.append(val)
  return li


# In[11]:


case_list1 = value_copy(case_list)
run_list1 = value_copy(run_list)
VB_list1 = value_copy(VB_list)


# In[12]:


time_list1 = value_copy(time_list)
DOC_list1 = value_copy(DOC_list)
feed_list1 = value_copy(feed_list)
material_list1 = value_copy(material_list)


# In[13]:


df = pd.DataFrame({'case':case_list1,'run':run_list1,'VB':VB_list1,'time':time_list1,'DOC':DOC_list1,'feed':feed_list1,'material':material_list1,'smc_AC':smc_AC_list,'smc_DC':smc_DC_list,'vib_table':vib_table_list,'vib_spindle':vib_spindle_list,'AE_table':AE_table_list,'AE_spindle':AE_spindle_list})


# In[14]:


df


# In[15]:


df.to_csv('mill_final.csv',index = False)


# In[15]:




