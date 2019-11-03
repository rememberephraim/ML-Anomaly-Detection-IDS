#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# !pip install brewer2mpl necessary for graphs
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Versions 
print(mpl.__version__)
print(sns.__version__)
print(np.__version__)


# In[4]:


import pandas as pd


# In[5]:


import plotly
import plotly.graph_objs as go
from chart_studio.plotly import plot, iplot
import chart_studio.plotly as py


# In[6]:


import re
from datetime import datetime 


# In[7]:


# Options for pandas
pd.options.display.max_columns = 88
# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[8]:


# Cufflinks wrapper on plotly
import cufflinks as cf


# In[9]:


from plotly.offline import iplot, init_notebook_mode, plot
cf.go_offline()
init_notebook_mode(connected=True)


# In[10]:


# Set global theme
cf.set_config_file(world_readable=True, theme='pearl')
import warnings  
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')


# In[11]:


#GETTING THE DATA FROM THE DATASET# 


# In[44]:


# Import the data
df = pd.read_csv('Syn.csv')
df.columns = df.columns.str.strip(" ").str.lower()
df.drop(['timestamp','destination ip','flow id','source ip','init_win_bytes_forward','init_win_bytes_backward','fwd urg flags','min_seg_size_forward','syn flag count','urg flag count','rst flag count','bwd urg flags','fin flag count','psh flag count','ece flag count','fwd avg bytes/bulk','fwd avg packets/bulk','fwd avg bulk rate','bwd avg bytes/bulk','bwd avg packets/bulk','bwd avg bulk rate'],axis=1,inplace=True)
df.sample(5)
df.corr()


# In[13]:


df.shape


# In[14]:


df['label'] = df['label'].map({'Syn':0,'BENIGN':1})
df['label'].value_counts()


# In[15]:


print('Syn', round(df['label'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('BENIGN', round(df['label'].value_counts()[1]/len(df) * 100,2), '% of the dataset')


# In[16]:


#Imbalance dataset
#New objective to solve the imbalance data set is to find a good
#-ration beteween SYN and Benign traffic .
#Hence we will look to split the dataframe into equal parts of SYN and Benign traffic

#For the data imbalance solution, we use the NearMiss and SMOTE Algorithm


# In[17]:


df.shape


# In[18]:


# number of missing values
df.isnull().sum()


# In[19]:


#Just to view some statistical values 
df.describe()


# In[20]:


#Exploratory Data Analysis


# In[21]:


#Looking at the number of unique values per feature
print(df.nunique())
#Looking at the top 2 rows
print(df.head(2))


# In[22]:


#For Faster Data Exploratory we can use pandas profiling.


# In[23]:


df.shape


# In[24]:


#Distributing
#This step we split the SYN and Benign data from the Dataset Equally.
#The reason is to reduce Bias of the algorithm 
#By reducing biasness we reduce overfitting and Wrong correlations


# In[25]:


df['label'].value_counts()


# In[26]:


# Shuffle the data before creating sub-samples
df = df.sample(frac=1)

#Here we applied the Near Miss algorithm, by making the majority class equal to minority class
df_syn = df.loc[df['label']==0][:392]
df_benign = df.loc[df['label']==1]

df_normal = pd.concat([df_syn,df_benign])

df_split  = df_normal.sample(frac=1,random_state=21)
df_split['label'].value_counts()


# In[27]:


df_split_corr = df_split.corr()
df_split_corr['label'].sort_values(ascending=False)


# In[28]:


#We find that the down/up ratio is top when it comes to distinguishing between Syn and Benign traffic


# In[29]:


#Building Models
from sklearn.model_selection import train_test_split
#importing the measurement metrics to be used 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler


# In[30]:


target_name = 'label'
X = df_split.drop('label', axis=1)
X = pd.get_dummies(X,prefix_sep='_',drop_first=True)
y = df_split[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123, stratify=y)


# In[31]:


def CMatrix(CM, labels=['Syn','Benign']):
    df = pd.DataFrame(data=CM, index=labels, columns=labels)
    df.index.name='TRUE'
    df.columns.name='PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df


# In[32]:


#Preparing a DataFrame for model analysis
# Data frame for evaluation metrics
metrics = pd.DataFrame(index=['accuracy', 'precision', 'recall'], 
                      columns=['LogisticReg', 'ClassTree', 'NaiveBayes'])


# In[33]:


#THERE WERE 3 DIFFERENT CLASSIFIERS THAT WERE USED: 
#-LOGISTIC REGRESSION
#-CLASSIFICATION TREE
#-NAIVE BAYES CLASSIFIERS 


# In[34]:


#Logistic Regression
#################################################################
# 1. Import the estimator object (model)
from sklearn.linear_model import LogisticRegression

# 2. Create an instance of the estimator
logistic_regression = LogisticRegression(n_jobs=-1, random_state=15)

# 3. Use the trainning data to train the estimator
logistic_regression.fit(X_train, y_train)

# 4. Evaluate the model
y_pred_test = logistic_regression.predict(X_test)
metrics.loc['accuracy','LogisticReg'] = accuracy_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision','LogisticReg'] = precision_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall','LogisticReg'] = recall_score(y_pred=y_pred_test, y_true=y_test)
#Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# In[35]:


#Classification Trees 
##################################################################
# 1. Import the estimator object (model)
from sklearn.tree import DecisionTreeClassifier

# 2. Create an instance of the estimator
class_tree = DecisionTreeClassifier(min_samples_split=30, min_samples_leaf=10, random_state=10)

# 3. Use the trainning data to train the estimator
class_tree.fit(X_train, y_train)

# 4. Evaluate the model
y_pred_test = class_tree.predict(X_test)
metrics.loc['accuracy','ClassTree'] = accuracy_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision','ClassTree'] = precision_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall','ClassTree'] = recall_score(y_pred=y_pred_test, y_true=y_test)
#Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# In[36]:


#Naive Bayes Classifier
# 1. Import the estimator object (model)
from sklearn.naive_bayes import GaussianNB

# 2. Create an instance of the estimator
NBC = GaussianNB()

# 3. Use the trainning data to train the estimator
NBC.fit(X_train, y_train)

# 4. Evaluate the model
y_pred_test = NBC.predict(X_test)
metrics.loc['accuracy','NaiveBayes'] = accuracy_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision','NaiveBayes'] = precision_score(y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall','NaiveBayes'] = recall_score(y_pred=y_pred_test, y_true=y_test)

#Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# In[37]:


#Here we want to compare the 3 different models 
100*metrics


# In[38]:


fig, ax = plt.subplots(figsize=(8,5))
metrics.plot(kind='barh', ax=ax)
ax.grid();


# In[39]:


#TESTING!!!!#######################################
precision_nb, recall_nb, thresholds_nb = precision_recall_curve(y_true=y_test, 
                                                                probas_pred=NBC.predict_proba(X_test)[:,1])
precision_lr, recall_lr, thresholds_lr = precision_recall_curve(y_true=y_test, 
                                                                probas_pred=logistic_regression.predict_proba(X_test)[:,1])                           


# In[40]:


fig, ax = plt.subplots(figsize=(8,5))
ax.plot(precision_nb, recall_nb, label='NaiveBayes')
ax.plot(precision_lr, recall_lr, label='LogisticReg')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precision-Recall Curve')
ax.hlines(y=0.5, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid();


# In[41]:


#Confusion matrix for modified Logistic Regression Classifier#
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(thresholds_lr, precision_lr[1:], label='Precision')
ax.plot(thresholds_lr, recall_lr[1:], label='Recall')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Precision, Recall')
ax.set_title('Logistic Regression Classifier: Precision-Recall')
ax.hlines(y=0.6, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid();


# In[42]:


#Classifier with threshold of 0.2
y_pred_proba = logistic_regression.predict_proba(X_test)[:,1]
y_pred_test = (y_pred_proba >= 0.3).astype('int')
#Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
print("Recall: ", 100*recall_score(y_pred=y_pred_test, y_true=y_test))
print("Precision: ", 100*precision_score(y_pred=y_pred_test, y_true=y_test))
CMatrix(CM)


# In[ ]:




