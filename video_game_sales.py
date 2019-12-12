#!/usr/bin/env python
# coding: utf-8

# In[1]:


#####################################
#              INST326              #
#             12/12/19              #
#              Team 12              #
#   John Henrici, Noah Engelmeyer,  #
# Aviva Moshman & Ethan Silberstein #
#           Final Project           #
#####################################

# import modules
import csv
import pandas as pd
import os
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# set working directory
os.chdir("/Users/johnhenrici/Documents/INST326")

# read csv file
df = pd.read_csv("vgsales.csv")


# In[2]:


# descriptive statistics for North American sales
df["NA_Sales"].describe()


# In[3]:


# descriptive statistics for European sales
df["EU_Sales"].describe()


# In[4]:


# descriptive statistics for Japanese sales
df["JP_Sales"].describe()


# In[5]:


# descriptive statistics for rest of world sales
df["Other_Sales"].describe()


# In[6]:


# descriptive statistics for global sales
df["Global_Sales"].describe()


# In[11]:


# T-test for North American sales
stats.ttest_ind(df["Global_Sales"], df["NA_Sales"])


# In[12]:


# T-test for European sales
stats.ttest_ind(df["Global_Sales"], df["EU_Sales"])


# In[13]:


# T-test for Japanese sales
stats.ttest_ind(df["Global_Sales"], df["JP_Sales"])


# In[14]:


# T-test for Other sales
stats.ttest_ind(df["Global_Sales"], df["Other_Sales"])


# In[15]:


# regression statistics for North American sales
lm = smf.ols(formula = "Global_Sales ~ NA_Sales", data = df).fit()
print(lm.summary())


# In[16]:


# regression statistics for European sales
lm = smf.ols(formula = "Global_Sales ~ EU_Sales", data = df).fit()
print(lm.summary())


# In[17]:


# regression statistics for Japanese sales
lm = smf.ols(formula = "Global_Sales ~ JP_Sales", data = df).fit()
print(lm.summary())


# In[18]:


# regression statistics of rest of world sales
lm = smf.ols(formula = "Global_Sales ~ Other_Sales", data = df).fit()
print(lm.summary())


# In[22]:


# regression plot of North American vs global sales
sns.regplot(x = "NA_Sales", y = "Global_Sales", data = df)


# In[23]:


# regression plot of European vs global sales
sns.regplot(x = "EU_Sales", y = "Global_Sales", data = df)


# In[24]:


# regression plot of Japanese vs global sales
sns.regplot(x = "JP_Sales", y = "Global_Sales", data = df)


# In[25]:


# regression plot of rest of world vs global sales
sns.regplot(x = "Other_Sales", y = "Global_Sales", data = df)


# In[ ]:




