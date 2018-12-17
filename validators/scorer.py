#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np


# In[9]:


def pad_remover(x):
    """ input is list from formatted df 
        output is the new list without -1"""
    x = set(x)
    if (-1 in x):
        x.remove(-1)
    if ("-1" in x):
        x.remove("-1")
    if ("NaN" in x):
        x.remove("NaN")
    if (np.nan in x):
        x.remove(np.nan)
    if ('nan' in x):
        x.remove('nan')
    x = list(x)
    return x


def penalty(public, answer):
    """
    penalty on base of the number of customer present in the answer"""
    a = set(public.CustomerID.unique())
    n = public.CustomerID.nunique()
    b = set(answer.CustomerID.unique())
    p = a.intersection(b)
    p = len(p)
    miss = (1-(p/n))*100
    if miss == 0:
        penalty = 0
    elif miss <= 10:
        penalty = 0.1
    elif miss <= 20:
        penalty = 0.2
    else:
        penalty = 0.4

    return penalty
        

def formatter(df):
    """input : recommended data frame
    
    formats it such a way that the item columns are stored in single column as a list
    """
    temp = df.apply(lambda x: x[1:].tolist(), axis=1)
    cust = df['CustomerID']
    df = pd.DataFrame({'CustomerID':cust, 'Items':temp})
    df.Items = df.Items.apply(lambda x: pad_remover(x))
    return df


def working_file(public, answer):
    """checks for id:
    gives inner left join over public data and customer data"""
    df = pd.merge(public,answer, on='CustomerID',how='left')
    return df


def scorer(df,df_public):
    """ generates score for a submission 
    """
    if len(df) == 0:
        return 0
    precision(df)
    recall(df)
    item_len(df)
    rec_len(df)
    df['fbeta']= df.apply(lambda x:Fbeta(x[3],x[4],x[5],x[6]),axis=1)
    score = df.fbeta.sum()/float(df_public.CustomerID.nunique())
    return score


def precision_helper(l1, l2):
    """
    l1: bought parameter
    l2: recomended parameter
    """
    a = set(l1)
    b = set(l2)
    pre = len(a.intersection(b))/len(b)
    return pre


def precision(df):
    df['precision'] = df.apply(lambda x: precision_helper(x[1],x[2]),axis=1)
    return df


def recall_helper(l1,l2):
    """
    l1: bought parameter
    l2: recomended parameter
    """
    a = set(l1)
    b = set(l2)
    ins = a.intersection(b)
    if len(a) > 10:
        div = 10
    else :
        div = len(a)
    recall = len(ins)/div
    return recall


def item_len(df):
    df['item_len'] = df.Items_x.apply(lambda x : len(x))
    return df['item_len']


def rec_len(df):
    df['rec_len'] = df.Items_y.apply(lambda x : len(x))
    return df['rec_len']


def recall(df):
    df['recall'] = df.apply(lambda x: recall_helper(x[1],x[2]),axis=1)
    return df


def Fbeta(precision,recall,item_len,rec_len):
    """returns fbeta score
    note : this contains a slight modified beta score to give more importance to precision"""
    if recall == 0 or precision == 0:
        return 0
    beta = (item_len / rec_len)**2
    fbeta = (1 + beta)* recall *  precision /(precision + beta *recall)
    return fbeta


# In[10]:


df_public = pd.read_csv('online_retail_test_public_rajesh.csv')
df_recommended = pd.read_csv('vit_recommend_v3.csv')
df_recommended.fillna('-1',inplace=True)
df_pub = formatter(df_public)
df_ans = formatter(df_recommended)
df = working_file(df_pub,df_ans)
df = df.dropna()


# In[11]:


penalty = penalty(df_pub,df_ans)


# In[12]:


score = scorer(df,df_public)


# In[13]:


print("the final score is :{}".format(score))


# In[14]:


precision_helper([1,2,3],['a','b'])


# In[15]:


precision_helper([1,2],[3,4])


# In[16]:


precision_helper([1,2,3,4,5],[2,3,4,5,6,7,8])


# In[17]:


precision_helper([1,2,3],[1,2,3])


# In[20]:


print(df.columns)


# In[ ]:




