#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('Airline_Delay_Cause.csv')


# In[3]:


data.head()


# In[4]:


data.isna().sum()


# In[5]:


data.dropna(inplace=True)


# In[6]:


data.isna().sum()


# In[7]:


data.info()


# In[8]:


data.drop(['carrier','carrier_name','airport','airport_name' ],axis=1, inplace=True)


# In[9]:


data.info()


# In[10]:


data


# In[11]:


#will erase this column, axis1 for columns and axis o for rows. inplace for overwrite
X = data.drop(['late_aircraft_ct'], axis=1, inplace = False)
y = data['late_aircraft_ct']


# In[12]:


data.head()


# In[13]:


X.head()


# In[14]:


y


# In[15]:


X.shape


# In[16]:


# Import Libraries
from sklearn.model_selection import train_test_split

#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# In[17]:


X_train.head()


# In[18]:


y_train.head()


# In[19]:


# Import Libraries to train the model after detrmining the X,Y 
from sklearn.linear_model import LinearRegression




# In[23]:


#Applying Linear Regression Model OBJECTabs

LinearRegressionModel = LinearRegression(fit_intercept=True,copy_X=True,n_jobs=-1)



# In[32]:


get_ipython().run_cell_magic('time', '', '\n# the most important step, i will always use the object LinearRegressionModel\n\nLinearRegressionModel.fit(X_train, y_train)\n# y=t1x1+t2x2+t3x3........t16x16 will make the first step building the equation\n#will ranomaize the weights 0,85x1, 0,12x2 .....\n\n')


# In[33]:


# print('Linear Regression Train Score is : ' , LinearRegressionModel.score(X_train, y_train))


# In[ ]:


#Calculating Details

#print('Linear Regression Test Score is : ' , LinearRegressionModel.score(X_test, y_test))
#print('Linear Regression Coef is : ' , LinearRegressionModel.coef_)
#print('Linear Regression intercept is : ' , LinearRegressionModel.intercept_)
#print('----------------------------------------------------')

#Calculating Prediction
y_pred = LinearRegressionModel.predict(X_test)
#print('Predicted Value for Linear Regression is : ' , y_pred[:10])


# In[36]:


y_pred = LinearRegressionModel.predict(X_test)
y_pred


# In[38]:


X_test.head()


# In[35]:


list(y_pred)


# In[40]:


list(y_test)


# In[41]:


#Import Libraries
from sklearn.metrics import mean_squared_error 
#----------------------------------------------------

#----------------------------------------------------
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)


# In[ ]:


0.000216 

