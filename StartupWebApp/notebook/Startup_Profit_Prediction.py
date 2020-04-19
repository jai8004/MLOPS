#!/usr/bin/env python
# coding: utf-8

# ## Profit Prediction For Startup Multi-Linear Regression 
# 
# We are given the Data Set which contains following features
# 
# 1. R&D Spend
# 2. Administration
# 3. Marketing Spend
# 4. State
# 5. Profit
# 
# We need to create a model which could predict the profit.

# In[45]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#import joblib 
from sklearn.externals import joblib


# In[11]:


dataset = pd.read_csv('50_Startups.csv')


# In[12]:


dataset.head()


# In[13]:


dataset.describe()


# In[14]:



dataset.isna().sum()


# In[15]:


#Converting Cateogrical variables to dummy variables

#We will have Dummy trap problem as all the 3 varaibles are multicollinear 
#So we must drop the last column  
dataset=pd.get_dummies(dataset,drop_first=True)


# In[16]:


dataset.corr()


# **As we see that all  columns are correlated with profit and can be used to predict the profit but we are not sure about the which feature are most suitable for predicting Profit**

# In[17]:


#Divding the dataset target varaible and feature variable 
y = dataset['Profit']
X = dataset[ ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida','State_New York' ] ]


# In[18]:


# dividing the target variable and feature variabled
X.head()


# In[19]:


y.head()


# In[20]:


# For using OLS we need to give bias also 
#y=bx+c
ones = np.ones((50,1))
X_new = np.append(arr=ones , values=X, axis=1  )
X_new


# In[21]:


model_ols = sm.OLS(endog=y ,  exog=X_new ).fit()


# In[22]:


model_ols.summary()


# **As we can see the x2,x3,x4,x5 have the P-Value more than the significance value(0.05) so we must remove them one by and one with the fact that there is no decrease in the Adj. R-squared**

# In[23]:


# removing x5
X_new = X_new[: ,0:5]
model_ols = sm.OLS(endog=y ,  exog=X_new ).fit()
model_ols.summary()


# In[24]:


# removing x4
X_new = X_new[: ,0:4]
model_ols = sm.OLS(endog=y ,  exog=X_new ).fit()
model_ols.summary()


# In[25]:


# removing x3
X_new = X_new[: ,0:3]
model_ols = sm.OLS(endog=y ,  exog=X_new ).fit()
model_ols.summary()


# In[26]:


# as we can see that Adjusted R-squared value also decrease 
# after removing the x3 vaiable so we must not proceed with removing it.

#Final Features 
X = dataset[ ['R&D Spend', 'Administration', 'Marketing Spend'] ]


# In[27]:


X.head()


# In[28]:


y.head()


# In[29]:


#dividing the training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=52)


# In[30]:


# creating our model
model = LinearRegression()


# In[31]:


#Training the model
model.fit(X_train , y_train)


# In[32]:


y_pred = model.predict(X_test)


# In[46]:


plt.scatter( y_test,y_pred)
sns.se
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Model Prediction')

plt.show()


# In[35]:


#finding the accuracy of the  model 
r2_score(y_test, y_pred)


# In[36]:


#saving our model 
joblib.dump(model, "profit_prediction.pk")


# In[44]:


#testing our dump model

RnD = 180380
Administration = 149487
Marketing  = 164701
test_feature= [RnD,Administration,Marketing]

test_feature_arr = np.array(test_feature)
test_feature_arr = test_feature_arr.reshape(1, -1)
profit_model = joblib.load("profit_prediction.pk")
profit_prediction = profit_model.predict(test_feature_arr )

print(profit_prediction[0])
print(str(round(float(profit_prediction), 2)))

