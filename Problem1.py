#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv('MBAAdmission/train.csv')
df.head()


# ### P1-1
# - 给出训练集数值型变量的count、mean、std、min、mid、max的统计特征
# - 给出训练集非数值型变量的count、unique、value的表

# In[2]:


print (df.describe())


# In[3]:


cols_notnum = df[['gender','international','major','race','work_industry','admission']]

for col in cols_notnum:
    value_cnt = df[col].value_counts()
    print (value_cnt)
    print(f"Count: {df[col].count()}")
    print(f"Unique values: {df[col].nunique()}")
    print ()
    


# ### P1-2
# - 对缺失值进行合适的处理，要求至少使用两种方法完成缺失值的补充；

# In[4]:


cols_num = ['application_id', 'gpa', 'gmat', 'work_exp']
cols_notnum = ['gender','international','major','race','work_industry','admission']


#数值用平均数， 非数值变量用众数：
def fill_method_1(df):  
    
    df_filled = df.copy()
    # numeric
    for col in cols_num:
        if df_filled[col].isnull().sum() > 0:
            mean = df[col].mean()
            df_filled[col] = df_filled[col].fillna(mean)
    
    for col in cols_notnum:
        if df[col].isnull().sum() > 0:
            mode_val = df_filled[col].mode()[0]
            df_filled[col] = df_filled[col].fillna(mode_val)
    
    return df_filled   

df_filled = fill_method_1(df)
print (type(df_filled))
print (df_filled.isnull().sum())


# ### P1-3
# - 将分类属性进行OneHot编码，你需要对Gender进行标签编码，对international for MBA进行除所述两个编码外的任意编码，其余需要进行编码的数据采取one-hot编码；

# In[5]:


from sklearn.preprocessing import LabelEncoder

col_without_two = ['major','race','work_industry']

def encode_data(df):
    df_encoded = df.copy()
    
    # label encoding
    if 'gender' in df.columns:
        le_gender = LabelEncoder() #创建编码器对象
        df_encoded['gender'] = le_gender.fit_transform(df['gender'].astype(str))
        
    #freq encoding
    if 'international' in df.columns:
        freq = df['international'].value_counts(normalize=True)
        df_encoded['international'] = df['international'].map(freq)
        
    if 'admission' in df.columns:
        le = LabelEncoder()
        df_encoded['admission'] = le.fit_transform(df['admission'])
    
    for col in col_without_two:
        dummies = pd.get_dummies(df_encoded[col], prefix=col, prefix_sep='_',
                                dummy_na=False)
        # 添加到编码后的DataFrame
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        # 移除原始列
        df_encoded.drop(col, axis=1, inplace=True)
    
    return df_encoded

df_encoded = encode_data(df_filled)

col_encoded = []
for col in df_encoded.columns:
    col_encoded.append(col)
print (col_encoded)
df_encoded.head()


# ### P1-4
# - 对数值属性进行必要的操作，如归一化处理等；

# In[6]:


from sklearn.preprocessing import StandardScaler

def standardize_numeric(df, cols_num):
    df_scaled = df.copy()
    scaler = StandardScaler()
    
    # 检查并处理缺失值
    for col in cols_num:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df_scaled[col] = df[col].fillna(median_val)
            print(f"列 {col} 的缺失值用中位数 {median_val:.2f} 填充")
    
    # 标准化
    df_scaled[cols_num] = scaler.fit_transform(df_scaled[cols_num])
    
    #print("\n标准化后的数据统计 (均值为0，标准差为1):")
    #print(df_scaled[cols_num].describe())
    
    return df_scaled, scaler

df_scaled, _ = standardize_numeric(df_encoded, cols_num)
df_scaled.head()


# ### P1-5
# - 可视化分析数据，展示数据分布，发现规律；

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. 数值变量的分布分析
def plot_numeric_distributions(df):
    """绘制数值变量的分布图"""
    numeric_cols = ['gpa', 'gmat', 'work_exp']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i >= len(axes):
            break
            
        # 箱线图
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'{col} - 箱线图')
        axes[i].set_ylabel('')
        
        # 直方图 + KDE
        if i + 3 < len(axes):
            sns.histplot(df[col], kde=True, ax=axes[i+3], bins=20)
            axes[i+3].set_title(f'{col} - 分布直方图')
            axes[i+3].set_xlabel('')
            
            # 添加统计信息
            stats_text = f'均值: {df[col].mean():.2f}\n中位数: {df[col].median():.2f}\n标准差: {df[col].std():.2f}'
            axes[i+3].text(0.05, 0.95, stats_text, transform=axes[i+3].transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 移除多余的子图
    for j in range(i+4, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle('数值变量分布分析', fontsize=16, y=1.02)
    plt.show()

plot_numeric_distributions(df)


# ### P2-1
# - 本题需要你分别用线性回归和逻辑回归对该数据集分类，请给出在训练集和测试集上的准确率。

# In[8]:


get_ipython().system(' jupyter nbconvert --to script Problem1.ipynb')



# In[ ]:




