# -*- coding: utf-8 -*-
"""DTW_dataset_chatter_detection_03_05_21.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Lr8G-l_15NUB1oR67NM_Ae1V5kAlqsWF

#Converting .mat file to pandas dataframe
Reference: https://towardsdatascience.com/how-to-load-matlab-mat-files-in-python-1f200e1287b5¶
"""

from google.colab import drive 
drive.mount('drive', force_remount = False)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import scipy
from sklearn import preprocessing
import random

def create_csv(path):
  data = loadmat(path)
  df = pd.DataFrame(data['tsDS'], columns = ['time', 'acc_x'])
  return df

"""#2.0 inch stickout distance"""

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_320_005.mat"
df_2p0_c_320_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_425_020.mat"
df_2p0_c_425_020 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_425_025.mat"
df_2p0_c_425_025 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_570_001.mat"
df_2p0_c_570_001 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_570_002.mat"
df_2p0_c_570_002 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_570_005.mat"
df_2p0_c_570_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_570_010.mat"
df_2p0_c_570_010 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_770_001.mat"
df_2p0_c_770_001 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_770_002.mat"
df_2p0_c_770_002 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_770_002_2.mat"
df_2p0_c_770_002_2 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_770_005.mat"
df_2p0_c_770_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/c_770_010.mat"
df_2p0_c_770_010 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/i_320_005.mat"
df_2p0_i_320_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/i_320_010.mat"
df_2p0_i_320_010 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/i_425_020.mat"
df_2p0_i_425_020 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/i_425_025.mat"
df_2p0_i_425_025 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/i_570_002.mat"
df_2p0_i_570_002 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/i_570_005.mat"
df_2p0_i_570_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/i_570_010.mat"
df_2p0_i_570_010 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/i_770_001.mat"
df_2p0_i_770_001 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_005.mat"
df_2p0_s_320_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_010.mat"
df_2p0_s_320_010 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_015.mat"
df_2p0_s_320_015 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_020.mat"
df_2p0_s_320_020 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_020_2.mat"
df_2p0_s_320_020_2 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_025.mat"
df_2p0_s_320_025 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_030.mat"
df_2p0_s_320_030 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_035.mat"
df_2p0_s_320_035 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_040.mat"
df_2p0_s_320_040 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_045.mat"
df_2p0_s_320_045 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_050.mat"
df_2p0_s_320_050 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_320_050_2.mat"
df_2p0_s_320_050_2 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_425_005.mat"
df_2p0_s_425_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_425_010.mat"
df_2p0_s_425_010 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_425_015.mat"
df_2p0_s_425_015 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_425_017.mat"
df_2p0_s_425_017 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_425_020.mat"
df_2p0_s_425_020 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_570_002.mat"
df_2p0_s_570_002 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/s_570_005.mat"
df_2p0_s_570_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_320_005.mat"
df_2p0_u_320_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_320_010.mat"
df_2p0_u_320_010 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_320_020.mat"
df_2p0_u_320_020 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_320_050.mat"
df_2p0_u_320_050 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_425_017.mat"
df_2p0_u_425_017 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_425_017_2.mat"
df_2p0_u_425_017_2 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_425_020.mat"
df_2p0_u_425_020 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_570_002.mat"
df_2p0_u_570_002 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_570_005.mat"
df_2p0_u_570_005 = create_csv(path)

path = "drive/MyDrive/cutting_tests_processed/2inch_stickout/u_770_002.mat"
df_2p0_u_770_002 = create_csv(path)

"""# 2.5 inch stickout distance

"""

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/c_570_014.mat"
# df_2p5_c_570_014 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/c_570_015s.mat"
# df_2p5_c_570_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/c_770_005.mat"
# df_2p5_c_770_005 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/i_570_012.mat"
# df_2p5_i_570_012 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/i_570_014.mat"
# df_2p5_i_570_014 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/i_570_015s.mat"
# df_2p5_i_570_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/i_770_005.mat"
# df_2p5_i_770_005 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/s_570_003.mat"
# df_2p5_s_570_003 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/s_570_005.mat"
# df_2p5_s_570_005 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/s_570_005_2.mat"
# df_2p5_s_570_005_2 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/s_570_008.mat"
# df_2p5_s_570_008 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/s_570_010.mat"
# df_2p5_s_570_010 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/s_570_015.mat"
# df_2p5_s_570_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/s_570_015_2.mat"
# df_2p5_s_570_015_2 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/s_770_002.mat"
# df_2p5_s_770_002 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/s_770_005.mat"
# df_2p5_s_770_005 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/u_570_005.mat"
# df_2p5_u_570_005 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/u_570_005_2.mat"
# df_2p5_u_570_005_2 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/u_570_015.mat"
# df_2p5_u_570_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/u_570_015_2.mat"
# df_2p5_u_570_015_2 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/u_570_015_3.mat"
# df_2p5_u_570_015_3 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/u_570_015_4.mat"
# df_2p5_u_570_015_4 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/2p5inch_stickout/u_770_002.mat"
# df_2p5_u_770_002 = create_csv(path)

"""# 3.5 inch stickout distance"""

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/c_1030_002.mat"
# df_3p5_c_1030_002 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/c_770_015.mat"
# df_3p5_c_770_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/i_770_010.mat"
# df_3p5_i_770_010 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/i_770_010_2.mat"
# df_3p5_i_770_010_2 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/i_770_015.mat"
# df_3p5_i_770_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/s_570_015.mat"
# df_3p5_s_570_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/s_570_025.mat"
# df_3p5_s_570_025 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/s_570_025_2.mat"
# df_3p5_s_570_025_2 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/s_570_030.mat"
# df_3p5_s_570_030 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/s_770_005.mat"
# df_3p5_s_770_005 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/s_770_008.mat"
# df_3p5_s_770_008 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/s_770_010.mat"
# df_3p5_s_770_010 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/s_770_010_2.mat"
# df_3p5_s_770_010_2 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/s_770_015.mat"
# df_3p5_s_770_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/u_570_015.mat"
# df_3p5_u_570_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/u_570_025.mat"
# df_3p5_u_570_025 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/u_570_030.mat"
# df_3p5_u_570_030 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/u_770_005.mat"
# df_3p5_u_770_005 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/3p5inch_stickout/u_770_015.mat"
# df_3p5_u_770_015 = create_csv(path)

"""# 4.5 inch stickout distance"""

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/c_1030_010.mat"
# df_4p5_c_1030_010 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/c_1030_015.mat"
# df_4p5_c_1030_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/c_1030_016.mat"
# df_4p5_c_1030_016 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/c_570_035.mat"
# df_4p5_c_570_035 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/c_570_040.mat"
# df_4p5_c_570_040 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/i_1030_010.mat"
# df_4p5_i_1030_010 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/i_1030_012.mat"
# df_4p5_i_1030_012 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/i_1030_013.mat"
# df_4p5_i_1030_013 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/i_1030_014.mat"
# df_4p5_i_1030_014 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_1030_005.mat"
# df_4p5_s_1030_005 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_1030_007.mat"
# df_4p5_s_1030_007 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_1030_013.mat"
# df_4p5_s_1030_013 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_1030_014.mat"
# df_4p5_s_1030_014 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_570_005.mat"
# df_4p5_s_570_005 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_570_010.mat"
# df_4p5_s_570_010 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_570_015.mat"
# df_4p5_s_570_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_570_025.mat"
# df_4p5_s_570_025 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_570_035.mat"
# df_4p5_s_570_035 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_570_040.mat"
# df_4p5_s_570_040 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_770_010.mat"
# df_4p5_s_770_010 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_770_015.mat"
# df_4p5_s_770_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/s_770_020.mat"
# df_4p5_s_770_020 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/u_570_015.mat"
# df_4p5_u_570_015 = create_csv(path)

# path = "drive/MyDrive/cutting_tests_processed/4p5inch_stickout/u_570_040.mat"
# df_4p5_u_570_040 = create_csv(path)

df_2p0_list = [df_2p0_c_320_005, df_2p0_c_425_020, df_2p0_c_425_025, df_2p0_c_570_001, df_2p0_c_570_002,
               df_2p0_c_570_005, df_2p0_c_570_010, df_2p0_c_770_001, df_2p0_c_770_002, df_2p0_c_770_002_2,
                df_2p0_c_770_005, df_2p0_c_770_010 , df_2p0_i_320_005, df_2p0_i_320_010, df_2p0_i_425_020,
               df_2p0_i_425_025, df_2p0_i_570_002, df_2p0_i_570_005, df_2p0_i_570_010, df_2p0_i_770_001,
              df_2p0_s_320_005, df_2p0_s_320_010, df_2p0_s_320_015, df_2p0_s_320_020, df_2p0_s_320_020_2,
               df_2p0_s_320_025, df_2p0_s_320_030, df_2p0_s_320_035, df_2p0_s_320_040, df_2p0_s_320_045,
                df_2p0_s_320_050, df_2p0_s_320_050_2, df_2p0_s_425_005, df_2p0_s_425_010, df_2p0_s_425_015,
               df_2p0_s_425_017, df_2p0_s_425_020, df_2p0_s_570_002, df_2p0_s_570_005, df_2p0_u_320_005,
                df_2p0_u_320_010, df_2p0_u_320_020, df_2p0_u_320_050, df_2p0_u_425_017, df_2p0_u_425_017_2,
               df_2p0_u_425_020, df_2p0_u_570_002, df_2p0_u_570_005, df_2p0_u_770_002]


# df_2p5_list = [df_2p5_c_570_014, df_2p5_c_570_015, df_2p5_c_770_005, df_2p5_i_570_012, df_2p5_i_570_014,
#                df_2p5_i_570_015, df_2p5_i_770_005, df_2p5_s_570_003, df_2p5_s_570_005, df_2p5_s_570_005_2,
#                df_2p5_s_570_008, df_2p5_s_570_010, df_2p5_s_570_015, df_2p5_s_570_015_2, df_2p5_s_770_002,
#                df_2p5_s_770_005, df_2p5_u_570_005, df_2p5_u_570_005_2, df_2p5_u_570_015, df_2p5_u_570_015_2,
#                df_2p5_u_570_015_3, df_2p5_u_570_015_4, df_2p5_u_770_002]

# df_3p5_list = [df_3p5_c_1030_002, df_3p5_c_770_015, df_3p5_i_770_010, df_3p5_i_770_010_2, df_3p5_i_770_015,
#                df_3p5_s_570_015, df_3p5_s_570_025, df_3p5_s_570_025_2, df_3p5_s_570_030, df_3p5_s_770_005,
#                df_3p5_s_770_008, df_3p5_s_770_010, df_3p5_s_770_010_2, df_3p5_s_770_015, df_3p5_u_570_015,
#                df_3p5_u_570_025, df_3p5_u_570_030, df_3p5_u_770_005, df_3p5_u_770_015]


# df_4p5_list = [df_4p5_c_1030_010, df_4p5_c_1030_015, df_4p5_c_1030_016, df_4p5_c_570_035, df_4p5_c_570_040, df_4p5_i_1030_010,
#                df_4p5_i_1030_013, df_4p5_i_1030_014, df_4p5_s_1030_007, df_4p5_s_1030_013, df_4p5_s_1030_014, df_4p5_s_570_005,
#                 df_4p5_s_570_010, df_4p5_s_570_015, df_4p5_s_570_025, df_4p5_s_570_035, df_4p5_s_570_040, df_4p5_s_770_010,
#                df_4p5_s_770_015, df_4p5_s_770_020, df_4p5_u_570_015, df_4p5_u_570_040]

def standardization(df):
  std = preprocessing.StandardScaler()
  for x in df.columns:
    if (x!='time'):
      df[x] = std.fit_transform(np.array(df[x]).reshape(-1,1))
  return df

def standardized_data(df_list):
  df_ans_list = []
  for i in range(len(df_list)):
    df = standardization(df_list[i])
    df_ans_list.append(df)
  return df_ans_list

# df_2p0_list[0]

df_2p0_list = standardized_data(df_2p0_list)

# df_2p5_list = standardized_data(df_2p5_list)

# df_3p5_list = standardized_data(df_3p5_list)

# df_4p5_list = standardized_data(df_4p5_list)

# df_2p0_list[0]

"""#LINE PLOT IN TIME DOMAIN"""

def line_plot(df, ti):
  plt.figure(figsize = (150,40))
  plt.plot(df['time'], df['acc_x'])
  plt.savefig(ti)

"""# FREQUENCY PLOT """

def time_2_freq_conversion(df, ti):
    Fx = np.array(df.acc_x)
    X = scipy.fft.fft(Fx)
    # print(X)
    N = len(X)
    # print(N)
    n = np.arange(N)
    # print(n)
    # get the sampling rate
    sr = 1 / (10000)
    # print(sr)
    T = N/sr
    # print(T)
    freq = n/T
    # print(freq) 

    # Get the one-sided specturm
    n_oneside = N//2
    # print(n_oneside)
    # get the one side frequency
    f_oneside = freq[:n_oneside]
    # print(f_oneside)

    plt.figure(figsize = (60, 10))
    plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    # plt.xlim(0, 0.5*(1e-7))
    # plt.ylim(0, 0.5*(1e7))
    st = "Frequency_"
    st+="acc_x"
    st+=" (Hz)"
    plt.xlabel(st)
    st = "FFT Amplitude |"
    st+= "acc_x"
    st+="(freq)|"
    plt.ylabel(st)
    plt.savefig(ti)

"""#Stickout distance = 2 inch, Cutting speed = 320 rpm, Depth of cut = 0.005 inch

##For Chatter
"""

# line_plot(df_2p0_c_320_005, "line_df_2p0_c_320_005")

# time_2_freq_conversion(df_2p0_c_320_005, "freq_df_2p0_c_320_005")

"""## For Intermediate chatter"""

# line_plot(df_2p0_i_320_005, "time_df_2p0_i_320_005")

# time_2_freq_conversion(df_2p0_i_320_005, "freq_df_2p0_i_320_005")

"""## For No chatter (stable region)"""

# line_plot(df_2p0_s_320_005, "line_df_2p0_s_320_005")

# time_2_freq_conversion(df_2p0_s_320_005, "freq_df_2p0_s_320_005")

"""## For Unknown region"""

# line_plot(df_2p0_u_320_005, "line_df_2p0_u_320_005")

# time_2_freq_conversion(df_2p0_u_320_005, "freq_df_2p0_u_320_005")

"""#Stickout distance = 2 inch, Cutting speed = 425 rpm, Depth of cut = 0.020 inch

##For Chatter
"""

# line_plot(df_2p0_c_425_020, "line_df_2p0_c_425_020")

# time_2_freq_conversion(df_2p0_c_425_020, "freq_df_2p0_c_425_020")

"""## For Intermediate chatter"""

# line_plot(df_2p0_i_425_020, "line_df_2p0_i_425_020")

# time_2_freq_conversion(df_2p0_i_425_020, "freq_df_2p0_i_425_020")

"""## For No chatter (stable region)"""

# line_plot(df_2p0_s_425_020, "line_df_2p0_s_425_020")

# time_2_freq_conversion(df_2p0_s_425_020, "freq_df_2p0_s_425_020")

"""## For Unknown region"""

# line_plot(df_2p0_u_425_020, "line_df_2p0_u_425_020")

# time_2_freq_conversion(df_2p0_u_425_020, "freq_df_2p0_u_425_020")

"""#Stickout distance = 2 inch, Cutting speed = 570 rpm, Depth of cut = 0.005 inch

##For Chatter
"""

# line_plot(df_2p0_c_570_005, "line_df_2p0_c_570_005")

# time_2_freq_conversion(df_2p0_c_570_005, "freq_df_2p0_c_570_005")

"""## For Intermediate chatter"""

# line_plot(df_2p0_i_570_005, "line_df_2p0_i_570_005")

# time_2_freq_conversion(df_2p0_i_570_005, "freq_df_2p0_i_570_005")

"""## For No chatter (stable region)"""

# line_plot(df_2p0_s_570_005, "line_df_2p0_s_570_005")

# time_2_freq_conversion(df_2p0_s_570_005, "freq_df_2p0_s_570_005")

"""## For Unknown region"""

# line_plot(df_2p0_u_570_005, "line_df_2p0_u_570_005")

# time_2_freq_conversion(df_2p0_u_570_005, "freq_df_2p0_u_570_005")

"""#Stickout distance = 2.5 inch, Cutting speed = 570 rpm, Depth of cut = 0.015 inch

##For Chatter
"""

# line_plot(df_2p5_c_570_015, "line_df_2p5_c_570_015")

# time_2_freq_conversion(df_2p5_c_570_015, "freq_df_2p5_c_570_015")

"""## For Intermediate chatter"""

# line_plot(df_2p5_i_570_015, "line_df_2p5_i_570_015")

# time_2_freq_conversion(df_2p5_i_570_015, "freq_df_2p5_i_570_015")

"""## For No chatter (stable region) [Part 1]"""

# line_plot(df_2p5_s_570_015, "line_df_2p5_s_570_015")

# time_2_freq_conversion(df_2p5_s_570_015, "freq_df_2p5_s_570_015")

"""## For No chatter (stable region) [Part 2]

---


"""

# line_plot(df_2p5_s_570_015_2, "line_df_2p5_s_570_015_2")

# time_2_freq_conversion(df_2p5_s_570_015_2, "freq_df_2p5_s_570_015_2")

"""## For Unknown region [Part 1]"""

# line_plot(df_2p5_u_570_015, "line_df_2p5_u_570_015")

# time_2_freq_conversion(df_2p5_u_570_015, "freq_df_2p5_u_570_015")

"""## For Unknown region [Part 2]"""

# line_plot(df_2p5_u_570_015_2, "line_df_2p5_u_570_015_2")

# time_2_freq_conversion(df_2p5_u_570_015_2, "freq_df_2p5_u_570_015_2")

"""## For Unknown region [Part 3]"""

# line_plot(df_2p5_u_570_015_3, "line_df_2p5_u_570_015_3")

# time_2_freq_conversion(df_2p5_u_570_015_3, "freq_df_2p5_u_570_015_3")

"""## For Unknown region [Part 4]"""

# line_plot(df_2p5_u_570_015_4, "line_df_2p5_u_570_015_4")

# time_2_freq_conversion(df_2p5_u_570_015_4, "freq_df_2p5_u_570_015_4")

"""#Stickout distance = 3.5 inch, Cutting speed = 770 rpm, Depth of cut = 0.015 inch

##For Chatter
"""

# line_plot(df_3p5_c_770_015, "line_df_3p5_c_770_015")

# time_2_freq_conversion(df_3p5_c_770_015, "freq_df_3p5_c_770_015")

"""## For Intermediate chatter"""

# line_plot(df_3p5_i_770_015, "line_df_3p5_i_770_015")

# time_2_freq_conversion(df_3p5_i_770_015, "freq_df_3p5_i_770_015")

"""## For No chatter (stable region)"""

# line_plot(df_3p5_s_770_015, "line_df_3p5_s_770_015")

# time_2_freq_conversion(df_3p5_s_770_015, "freq_df_3p5_s_770_015")

"""## For Unknown region"""

# line_plot(df_3p5_u_770_015, "line_df_3p5_u_770_015")

# time_2_freq_conversion(df_3p5_u_770_015, "freq_df_3p5_u_770_015")

"""#Stickout distance = 4.5 inch, Cutting speed = 1030 rpm, Depth of cut = 0.010 inch

##For Chatter
"""

# line_plot(df_4p5_c_1030_010, "line_df_4p5_c_1030_010")

# time_2_freq_conversion(df_4p5_c_1030_010, "freq_df_4p5_c_1030_010")

"""## For Intermediate chatter"""

# line_plot(df_4p5_i_1030_010, "line_df_4p5_i_1030_010")

# time_2_freq_conversion(df_4p5_i_1030_010, "freq_df_4p5_i_1030_010")

"""# Train Test Split"""

def train_test_generator(df_list, test_num, val_num):
  train_list = []
  val_list = []
  test_list = []
  for i in range(len(df_list)):
    if i in test_num:
      test_list.append(df_list[i])
    elif i in val_num:
      val_list.append(df_list[i])
    else:
      train_list.append(df_list[i])
  return train_list, val_list, test_list

test_num = [4, 9, 15, 22, 26, 31, 33, 35, 36, 40]
val_num = [7, 17, 30, 37]

train_df_2p0_list, val_df_2p0_list, test_df_2p0_list = train_test_generator(df_2p0_list, test_num, val_num)

# print(len(train_df_2p0_list))
# print(len(test_df_2p0_list))
# print(len(val_df_2p0_list))

# print(df_2p0_list[4])
# print(train_df_2p0_list[4])
# print(test_df_2p0_list[0])

pip install dtw-python

from dtw import *

def dtw_submatrix(df1, df2):

  ######### df1 ##########

  length = len(df1)

  n1 = length//50
  n2 = n1 + n1
  n3 = n2 + n1
  n4 = n3 + n1
  n5 = n4 + n1
  n6 = n5 + n1
  n7 = n6 + n1
  n8 = n7 + n1
  n9 = n8 + n1
  n10 = n9 + n1

  n11 = n10 + n1
  n12 = n11 + n1
  n13 = n12 + n1
  n14 = n13 + n1
  n15 = n14 + n1
  n16 = n15 + n1
  n17 = n16 + n1
  n18 = n17 + n1
  n19 = n18 + n1
  n20 = n19 + n1

  n21 = n20 + n1
  n22 = n21 + n1
  n23 = n22 + n1
  n24 = n23 + n1
  n25 = n24 + n1
  n26 = n25 + n1
  n27 = n26 + n1
  n28 = n27 + n1
  n29 = n28 + n1
  n30 = n29 + n1

  n31 = n30 + n1
  n32 = n31 + n1
  n33 = n32 + n1
  n34 = n33 + n1
  n35 = n34 + n1
  n36 = n35 + n1
  n37 = n36 + n1
  n38 = n37 + n1
  n39 = n38 + n1
  n40 = n39 + n1

  n41 = n40 + n1
  n42 = n41 + n1
  n43 = n42 + n1
  n44 = n43 + n1
  n45 = n44 + n1
  n46 = n45 + n1
  n47 = n46 + n1
  n48 = n47 + n1
  n49 = n48 + n1
  


  df1_sub1 = df1.iloc[:n1,:]
  df1_sub2 = df1.iloc[n1:n2,:]
  df1_sub3 = df1.iloc[n2:n3,:]
  df1_sub4 = df1.iloc[n3:n4,:]
  df1_sub5 = df1.iloc[n4:n5,:]
  df1_sub6 = df1.iloc[n5:n6,:]
  df1_sub7 = df1.iloc[n6:n7,:]
  df1_sub8 = df1.iloc[n7:n8,:]
  df1_sub9 = df1.iloc[n8:n9,:]

  df1_sub10 = df1.iloc[n9:n10,:]
  df1_sub11 = df1.iloc[n10:n11,:]
  df1_sub12 = df1.iloc[n11:n12,:]
  df1_sub13 = df1.iloc[n12:n13,:]
  df1_sub14 = df1.iloc[n13:n14,:]
  df1_sub15 = df1.iloc[n14:n15,:]
  df1_sub16 = df1.iloc[n15:n16,:]
  df1_sub17 = df1.iloc[n16:n17,:]
  df1_sub18 = df1.iloc[n17:n18,:]
  df1_sub19 = df1.iloc[n18:n19,:]

  df1_sub20 = df1.iloc[n19:n20,:]
  df1_sub21 = df1.iloc[n20:n21,:]
  df1_sub22 = df1.iloc[n21:n22,:]
  df1_sub23 = df1.iloc[n22:n23,:]
  df1_sub24 = df1.iloc[n23:n24,:]
  df1_sub25 = df1.iloc[n24:n25,:]
  df1_sub26 = df1.iloc[n25:n26,:]
  df1_sub27 = df1.iloc[n26:n27,:]
  df1_sub28 = df1.iloc[n27:n28,:]
  df1_sub29 = df1.iloc[n28:n29,:]

  df1_sub30 = df1.iloc[n29:n30,:]
  df1_sub31 = df1.iloc[n30:n31,:]
  df1_sub32 = df1.iloc[n31:n32,:]
  df1_sub33 = df1.iloc[n32:n33,:]
  df1_sub34 = df1.iloc[n33:n34,:]
  df1_sub35 = df1.iloc[n34:n35,:]
  df1_sub36 = df1.iloc[n35:n36,:]
  df1_sub37 = df1.iloc[n36:n37,:]
  df1_sub38 = df1.iloc[n37:n38,:]
  df1_sub39 = df1.iloc[n38:n39,:]

  df1_sub40 = df1.iloc[n39:n40,:]
  df1_sub41 = df1.iloc[n40:n41,:]
  df1_sub42 = df1.iloc[n41:n42,:]
  df1_sub43 = df1.iloc[n42:n43,:]
  df1_sub44 = df1.iloc[n43:n44,:]
  df1_sub45 = df1.iloc[n44:n45,:]
  df1_sub46 = df1.iloc[n45:n46,:]
  df1_sub47 = df1.iloc[n46:n47,:]
  df1_sub48 = df1.iloc[n47:n48,:]
  df1_sub49 = df1.iloc[n48:n49,:]
  df1_sub50 = df1.iloc[n49:,:]

  df1_sublist = [ df1_sub1, df1_sub2, df1_sub3, df1_sub4, df1_sub5, df1_sub6,
          df1_sub7, df1_sub8, df1_sub9, df1_sub10, df1_sub11, df1_sub12, df1_sub13, 
          df1_sub14, df1_sub15, df1_sub16, df1_sub17, df1_sub18, df1_sub19, df1_sub20,
          df1_sub21, df1_sub22, df1_sub23, df1_sub24, df1_sub25, df1_sub26, df1_sub27, df1_sub28,
          df1_sub29, df1_sub30, df1_sub31, df1_sub32, df1_sub33, df1_sub34, df1_sub35, df1_sub36, 
          df1_sub37, df1_sub38, df1_sub39, df1_sub40, df1_sub41, df1_sub42, df1_sub43, df1_sub44, 
          df1_sub45, df1_sub46, df1_sub47, df1_sub48, df1_sub49, df1_sub50 ]
  
  


######### df2 ##########

  length = len(df2)

  n1 = length//50
  n2 = n1 + n1
  n3 = n2 + n1
  n4 = n3 + n1
  n5 = n4 + n1
  n6 = n5 + n1
  n7 = n6 + n1
  n8 = n7 + n1
  n9 = n8 + n1
  n10 = n9 + n1

  n11 = n10 + n1
  n12 = n11 + n1
  n13 = n12 + n1
  n14 = n13 + n1
  n15 = n14 + n1
  n16 = n15 + n1
  n17 = n16 + n1
  n18 = n17 + n1
  n19 = n18 + n1
  n20 = n19 + n1

  n21 = n20 + n1
  n22 = n21 + n1
  n23 = n22 + n1
  n24 = n23 + n1
  n25 = n24 + n1
  n26 = n25 + n1
  n27 = n26 + n1
  n28 = n27 + n1
  n29 = n28 + n1
  n30 = n29 + n1

  n31 = n30 + n1
  n32 = n31 + n1
  n33 = n32 + n1
  n34 = n33 + n1
  n35 = n34 + n1
  n36 = n35 + n1
  n37 = n36 + n1
  n38 = n37 + n1
  n39 = n38 + n1
  n40 = n39 + n1

  n41 = n40 + n1
  n42 = n41 + n1
  n43 = n42 + n1
  n44 = n43 + n1
  n45 = n44 + n1
  n46 = n45 + n1
  n47 = n46 + n1
  n48 = n47 + n1
  n49 = n48 + n1
  

  df2_sub1 = df2.iloc[:n1,:]
  df2_sub2 = df2.iloc[n1:n2,:]
  df2_sub3 = df2.iloc[n2:n3,:]
  df2_sub4 = df2.iloc[n3:n4,:]
  df2_sub5 = df2.iloc[n4:n5,:]
  df2_sub6 = df2.iloc[n5:n6,:]
  df2_sub7 = df2.iloc[n6:n7,:]
  df2_sub8 = df2.iloc[n7:n8,:]
  df2_sub9 = df2.iloc[n8:n9,:]

  df2_sub10 = df2.iloc[n9:n10,:]
  df2_sub11 = df2.iloc[n10:n11,:]
  df2_sub12 = df2.iloc[n11:n12,:]
  df2_sub13 = df2.iloc[n12:n13,:]
  df2_sub14 = df2.iloc[n13:n14,:]
  df2_sub15 = df2.iloc[n14:n15,:]
  df2_sub16 = df2.iloc[n15:n16,:]
  df2_sub17 = df2.iloc[n16:n17,:]
  df2_sub18 = df2.iloc[n17:n18,:]
  df2_sub19 = df2.iloc[n18:n19,:]

  df2_sub20 = df2.iloc[n19:n20,:]
  df2_sub21 = df2.iloc[n20:n21,:]
  df2_sub22 = df2.iloc[n21:n22,:]
  df2_sub23 = df2.iloc[n22:n23,:]
  df2_sub24 = df2.iloc[n23:n24,:]
  df2_sub25 = df2.iloc[n24:n25,:]
  df2_sub26 = df2.iloc[n25:n26,:]
  df2_sub27 = df2.iloc[n26:n27,:]
  df2_sub28 = df2.iloc[n27:n28,:]
  df2_sub29 = df2.iloc[n28:n29,:]

  df2_sub30 = df2.iloc[n29:n30,:]
  df2_sub31 = df2.iloc[n30:n31,:]
  df2_sub32 = df2.iloc[n31:n32,:]
  df2_sub33 = df2.iloc[n32:n33,:]
  df2_sub34 = df2.iloc[n33:n34,:]
  df2_sub35 = df2.iloc[n34:n35,:]
  df2_sub36 = df2.iloc[n35:n36,:]
  df2_sub37 = df2.iloc[n36:n37,:]
  df2_sub38 = df2.iloc[n37:n38,:]
  df2_sub39 = df2.iloc[n38:n39,:]

  df2_sub40 = df2.iloc[n39:n40,:]
  df2_sub41 = df2.iloc[n40:n41,:]
  df2_sub42 = df2.iloc[n41:n42,:]
  df2_sub43 = df2.iloc[n42:n43,:]
  df2_sub44 = df2.iloc[n43:n44,:]
  df2_sub45 = df2.iloc[n44:n45,:]
  df2_sub46 = df2.iloc[n45:n46,:]
  df2_sub47 = df2.iloc[n46:n47,:]
  df2_sub48 = df2.iloc[n47:n48,:]
  df2_sub49 = df2.iloc[n48:n49,:]
  df2_sub50 = df2.iloc[n49:,:]


  df2_row11 = [df2_sub1, df2_sub2, df2_sub3, df2_sub4, df2_sub5]
  df2_row12 = [df2_sub6, df2_sub7, df2_sub8, df2_sub9, df2_sub10]
  df2_row21 = [df2_sub11, df2_sub12, df2_sub13, df2_sub14, df2_sub15]
  df2_row22 = [df2_sub16, df2_sub17, df2_sub18, df2_sub19, df2_sub20]
  df2_row31 = [df2_sub21, df2_sub22, df2_sub23, df2_sub24, df2_sub25]
  df2_row32 = [df2_sub26, df2_sub27, df2_sub28, df2_sub29, df2_sub30]
  df2_row41 = [df2_sub31, df2_sub32, df2_sub33, df2_sub34, df2_sub35] 
  df2_row42 = [df2_sub36, df2_sub37, df2_sub38, df2_sub39, df2_sub40]
  df2_row51 = [df2_sub41, df2_sub42, df2_sub43, df2_sub44, df2_sub45]
  df2_row52 = [df2_sub46, df2_sub47, df2_sub48, df2_sub49, df2_sub50]

  df2_sublist = [df2_row11, df2_row12, df2_row21, df2_row22, df2_row31, df2_row32, df2_row41, df2_row42, df2_row51, df2_row52]

  return df1_sublist, df2_sublist

def dtw_submatrix_distance(df1, df2_list):
  lis = []
  for i in range(len(df2_list)):
    dtw_cal = dtw(df1, df2_list[i])
    lis.append(dtw_cal.distance)
  print(lis)
  return lis

def dtw_distance (df1_sublist, df2_sublist):
  lis = []
  for i in range(len(df1_sublist)):
    lis_sub1 = []
    for j in range(len(df2_sublist)):
      lis_sub = dtw_submatrix_distance (df1_sublist[i], df2_sublist[j])
      lis_sub1 += lis_sub
    lis.append(lis_sub1)
  print(lis)
  return lis

def dtw_general(df, ind, start_ind, end_ind):
  lis_sub = []
  for j in range(start_ind, end_ind):
    df1_sublist, df2_sublist = dtw_submatrix(df[ind], df[j])
    lis = dtw_distance (df1_sublist, df2_sublist)
    lis_sub.append(lis)
  return lis_sub

"""# 2.0 inches stickout distance matrix"""

def dtw_matrix_generator_func(train_df_2p0_list, ind, start_ind, end_ind):
  dtw_2p0_0 = dtw_general(train_df_2p0_list, ind, start_ind, end_ind)
  return dtw_2p0_0

# dtw_2p0_11_1 = dtw_matrix_generator_func(train_df_2p0_list, 11, 0, 5)
# with open("data_11_1.txt","w") as f:
#   for ele in dtw_2p0_11_1:
#     f.write("%s\n" %ele)

# dtw_2p0_11_2 = dtw_matrix_generator_func(train_df_2p0_list, 11, 5, 10)
# with open("data_11_2.txt","w") as f:
#   for ele in dtw_2p0_11_2:
#     f.write("%s\n" %ele)

# dtw_2p0_11_3 = dtw_matrix_generator_func(train_df_2p0_list, 11, 10, 15)
# with open("data_11_3.txt","w") as f:
#   for ele in dtw_2p0_11_3:
#     f.write("%s\n" %ele)

# dtw_2p0_11_4 = dtw_matrix_generator_func(train_df_2p0_list, 11, 15, 20)
# with open("data_11_4.txt","w") as f:
#   for ele in dtw_2p0_11_4:
#     f.write("%s\n" %ele)

# dtw_2p0_26_5 = dtw_matrix_generator_func(train_df_2p0_list, 26, 20, 25)
# with open("data_26_5.txt","w") as f:
#   for ele in dtw_2p0_26_5:
#     f.write("%s\n" %ele)

# dtw_2p0_26_6 = dtw_matrix_generator_func(train_df_2p0_list, 26, 25, 30)
# with open("data_26_6.txt","w") as f:
#   for ele in dtw_2p0_26_6:
#     f.write("%s\n" %ele)

dtw_2p0_28_7 = dtw_matrix_generator_func(train_df_2p0_list, 28, 30, 35)
with open("data_28_7.txt","w") as f:
  for ele in dtw_2p0_28_7:
    f.write("%s\n" %ele)

# dtw_2p0_train = []
  
  # dtw_2p0_0 += dtw_2p0_0_1

  # dtw_2p0_0_2 = dtw_general(train_df_2p0_list, ind, 5, 10)
  # dtw_2p0_0 += dtw_2p0_0_2

  # dtw_2p0_0_3 = dtw_general(train_df_2p0_list, ind, 10, 15)
  # dtw_2p0_0 += dtw_2p0_0_3

  # dtw_2p0_0_4 = dtw_general(train_df_2p0_list, ind, 15, 20)
  # dtw_2p0_0 += dtw_2p0_0_4

  # dtw_2p0_0_5 = dtw_general(train_df_2p0_list, ind, 20, 25)
  # dtw_2p0_0 += dtw_2p0_0_5

  # dtw_2p0_0_6 = dtw_general(train_df_2p0_list, ind, 25, 30)
  # dtw_2p0_0 += dtw_2p0_0_6

  # dtw_2p0_0_7 = dtw_general(train_df_2p0_list, ind, 30, 35)
  # dtw_2p0_0 += dtw_2p0_0_7
  
  # return dtw_2p0_0

"""# 2.5 inches stickout distance matrix"""

# lis_dtw_2p5 = dtw_general(df_2p5_list)

"""# 3.5 inches stickout distance matrix"""

# lis_dtw_3p5 = dtw_general(df_3p5_list)

"""# 4.5 inches stickout distance matrix"""

# lis_dtw_4p5 = dtw_general(df_4p5_list)

# def helper_func(df):
#   lis = []
#   for i in range(len(df)):
#     for j in range(len(df)):
#       lis.append((i,j))
#   return lis

# lis = helper_func(df_4p5_list)

# count = 0
# for i in range(len(lis)):
#   print(f'{lis[i][0]} and {lis[i][1]}')
#   print(f'{len(df_2p0_list[lis[i][0]])} and {len(df_2p0_list[lis[i][1]])}')
#   print(count)
#   count+=1
# print(count)
# print(len(lis))



"""# taking stable as the gold standard is what sir suggested"""

# def dtw_general(df):
#   lis_gen = []
#   for i in range(len(df)):
#     lis_sub = []
#     for j in range(len(df)):
#       df1_sublist, df2_sublist = dtw_submatrix(df[i], df[j])
#       lis = dtw_distance (df1_sublist, df2_sublist)
#       lis_sub.append(lis)
#     lis_gen.append(lis_sub)
#   return lis_gen