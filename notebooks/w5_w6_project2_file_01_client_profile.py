#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load the dataset, concatenate the 2-part digital_footprint file.
df_experiment_roster = pd.read_csv("/Users/milenko/My Drive (1307mile@gmail.com)/bootcamp/w5/w5w6_project2/data/raw/df_final_experiment_clients.txt")
df_client_profiles = pd.read_csv("/Users/milenko/My Drive (1307mile@gmail.com)/bootcamp/w5/w5w6_project2/data/raw/df_final_demo.txt")

df_digital_footprint_1 = pd.read_csv("/Users/milenko/My Drive (1307mile@gmail.com)/bootcamp/w5/w5w6_project2/data/raw/df_final_web_data_pt_1.txt")
df_digital_footprint_2 = pd.read_csv("/Users/milenko/My Drive (1307mile@gmail.com)/bootcamp/w5/w5w6_project2/data/raw/df_final_web_data_pt_2.txt")
df_digital_footprint = pd.concat([df_digital_footprint_1, df_digital_footprint_2], axis=0, ignore_index=True)


# # 1. Dataset exploration

# ## 1.1. df_experiment_roster

# In[3]:


df_experiment_roster


# In[4]:


df_experiment_roster.info()


# In[5]:


# Drop clients for which the Variation is unknown.
print(df_experiment_roster.shape)
df_experiment_roster = df_experiment_roster[df_experiment_roster['Variation'].isna()==False]
print(df_experiment_roster.shape)


# ## 1.2. df_client_profiles

# In[6]:


df_client_profiles


# In[7]:


df_client_profiles.info()


# In[8]:


# Drop 'clnt_tenure_mnth' for redundancy with 'clnt_tenure_yr'
df_client_profiles = df_client_profiles.drop(columns=['clnt_tenure_mnth'])


# In[9]:


# Drop the rows without demographic information.
print(df_client_profiles.shape)
df_client_profiles = df_client_profiles.dropna(thresh=8)
print(df_client_profiles.shape)


# In[10]:


# Convert floats to int
int_columns = ['clnt_tenure_yr', 'num_accts', 'calls_6_mnth', 'logons_6_mnth']

for col in int_columns:
    df_client_profiles[col] = df_client_profiles[col].astype(int).copy()


# In[11]:


# Merge df_client_profiles and df_experiment_roster
df_client_profiles = pd.merge(df_client_profiles, df_experiment_roster, on='client_id')
df_client_profiles


# In[12]:


# Rename columns for clarity and consistence.
df_client_profiles = df_client_profiles.rename(columns= {
    'clnt_tenure_yr': 'client_tenure_in_years', 
    'clnt_age': 'client_age', 
    'gendr': 'gender', 
    'num_accts': 'number_of_accounts', 
    'bal': 'balance',
    'Variation': 'experiment_group'})


# In[13]:


# Transform and rename 2 columns from 6-monthly to annual values.
df_client_profiles[['calls_6_mnth', 'logons_6_mnth']] = df_client_profiles[['calls_6_mnth', 'logons_6_mnth']].apply(lambda x: x * 2)
df_client_profiles.rename(columns= {'calls_6_mnth': 'calls_per_year', 'logons_6_mnth': 'logons_per_year'}, inplace=True)


# In[14]:


# Check for duplicates and nulls.

print(f"duplicated rows: {df_client_profiles.duplicated().sum()}")
print(f"duplicated client_id': {df_client_profiles['client_id'].duplicated().sum()}")
print(f"null values: {df_client_profiles.isna().sum().sum()}")


# ## 1.3. df_digital_footprint

# In[34]:


# Put aside this df for now. df_digital_footprint has 750k rows, and df_client_profile has 50k.
# Merging them would inflate attribute values and give inaccurate distributions.

df_digital_footprint


# In[16]:


df_digital_footprint.info()


# # 2. Client behaviour analysis
# - Who are the primary clients using this online process?
# - Are the primary clients younger or older, new or long-standing?

# In[37]:


df_client_profiles['dummy'] = 1

client_profiles_pivot = pd.pivot_table(df_client_profiles, 
                                      values=['client_tenure_in_years', 'client_age', 'number_of_accounts', 'balance', 'calls_per_year', 'logons_per_year'], 
                                      aggfunc={'client_tenure_in_years': 'mean',
                                               'client_age': 'mean', 
                                               'number_of_accounts': 'mean', 
                                               'balance': 'mean', 
                                               'calls_per_year': 'mean', 
                                               'logons_per_year': 'mean'},
                                      index='dummy')

client_profiles_pivot


# In[18]:


gender_value_counts = df_client_profiles['gender'].value_counts()
gender_value_counts


# # Who are the primary clients using this online process?
# 
# Back to this question.\
# If my primary objective is to decode the experiment’s performance, then my focus is on the Test group clients.

# In[19]:


test_group_pivot = pd.pivot_table(df_client_profiles[df_client_profiles['experiment_group'] == 'Test'], 
            index=['experiment_group'], 
            values=['client_tenure_in_years', 'client_age', 'number_of_accounts', 'balance', 'calls_per_year', 'logons_per_year'], 
            aggfunc={'client_tenure_in_years': 'mean',
                    'client_age': 'mean', 
                    'number_of_accounts': 'mean', 
                    'balance': 'mean', 
                    'calls_per_year': 'mean', 
                    'logons_per_year': 'mean'})
test_group_pivot


# In[20]:


test_gender_value_counts = df_client_profiles[df_client_profiles['experiment_group'] == 'Test']['gender'].value_counts()
test_gender_value_counts


# # Grouping for pivot
# New categories created from grouping values in existing columns.

# In[21]:


# tenure_group
tenure_bins = [0, 10, 20, 30, 50, float('inf')]
tenure_labels = ['0-10 years', '10-20 years', '20-30 years', '30-40 years', '40+ years']
df_client_profiles['tenure_group'] = pd.cut(df_client_profiles['client_tenure_in_years'], 
                                            bins=tenure_bins, labels=tenure_labels, right=False)

# age_group
age_bins = [0, 20, 40, 60, 80, float('inf')]
age_labels = ['0-20 years', '21-40 years', '41-60 years', '61-80 years', '80+ years']
df_client_profiles['age_group'] = pd.cut(df_client_profiles['client_age'], bins=age_bins, labels=age_labels, right=False)

# balance_group
balance_bins = [0, 200000, 400000, 600000, 800000, float('inf')]
balance_labels = ['0 - 200k', '200,001 - 400k', '400,001 - 600k', '600,001 - 800k', '800,001+']
df_client_profiles['balance_group'] = pd.cut(df_client_profiles['balance'], bins=balance_bins, labels=balance_labels, right=False)


# # Tenure group
# - **30-40: highest balance, makes most calls and logons per year, and have most accounts.**

# In[22]:


tenure_groups_pivot = pd.pivot_table(df_client_profiles[df_client_profiles['experiment_group'] == 'Test'], 
            observed=False,
            index=['tenure_group'], 
            values=['client_age', 'number_of_accounts', 'balance', 'calls_per_year', 'logons_per_year'], 
            aggfunc={'client_age': 'mean', 
                     'number_of_accounts': 'mean', 
                     'balance': 'mean',
                     'calls_per_year': 'mean', 
                     'logons_per_year': 'mean'})
display(tenure_groups_pivot)


# # Age group
# - **21-40: has most accounts.**
# - **61-80: highest balance, made most calls.**
# - **80+ made most logons.**

# In[23]:


age_groups_pivot = pd.pivot_table(df_client_profiles[df_client_profiles['experiment_group'] == 'Test'], 
            observed=False,
            index=['age_group'], 
            values=['client_tenure_in_years', 'number_of_accounts', 'balance', 'calls_per_year', 'logons_per_year'], 
            aggfunc={'client_tenure_in_years': 'mean',
                     'number_of_accounts': 'mean', 
                     'balance': 'mean', 
                     'calls_per_year': 'mean', 
                     'logons_per_year': 'mean'})
display(age_groups_pivot)


# # Balance group
# - **800k+ tops every attribute.**

# In[24]:


balance_groups_pivot = pd.pivot_table(df_client_profiles[df_client_profiles['experiment_group'] == 'Test'], 
            observed=False,
            index=['balance_group'], 
            values=['client_tenure_in_years', 'client_age', 'number_of_accounts', 'calls_per_year', 'logons_per_year'], 
            aggfunc={'client_tenure_in_years': 'mean',
                     'client_age': 'mean', 
                     'number_of_accounts': 'mean', 
                     'calls_per_year': 'mean', 
                     'logons_per_year': 'mean'})
display(balance_groups_pivot)


# In[31]:


df_client_profiles.to_csv("/Users/milenko/My Drive (1307mile@gmail.com)/bootcamp/w5/w5w6_project2/data/final/df_client_profiles_final.csv", index=False)
df_digital_footprint.to_csv("/Users/milenko/My Drive (1307mile@gmail.com)/bootcamp/w5/w5w6_project2/data/final/df_digital_footprint_final.csv", index=False)

