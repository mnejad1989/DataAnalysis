#!/usr/bin/env python
# coding: utf-8

# # Building Machine Learning Pipelines: Data Analysis Phase
# 
# Here we will focus on creating Machine Learning Pipelines considering all the life cycle of a Data Science Projects.

# # Project Name: House Prices: Advanced Regression Techniques
# 
# The main aim of this project is to predict the house price based on various features which we will discuss as we go ahead

# # Dataset to downloaded from the below link
# 
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# # All the Lifecycle In A Data Science Projects
# 
# 1. Data Analysis. Or Data Preprocessing
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 5. Model Deployment

# In[187]:


## Data Analysis Phase
## MAin aim is to understand more about the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[188]:


# Display all the columns of the dataframe

pd.pandas.set_option('display.max_columns',None)


# In[189]:


dataset=pd.read_csv('House Prices/train.csv')


# In[190]:


dataset.head()


# In[191]:


## print shape of the dataset with rows and columns
print(dataset.shape)


# In Data Analysis We will Analyze To Find out the below stuff
# 
# 1. Missing Values
# 2. All The Numerical Variables
# 3. Distribution of the Numerical Variables
# 4. Categorical Variables
# 5. Cardinality of Categorical Variables
# 6. Outliers
# 7. Relationship between independent and dependent feature(SalePrice)

# # 1. Missing Values

# ### Here we will check the percentage of nan values present in each feature
# ### Step 1 - make the list of features which has missing values
# 
# In feature engineering we will handle the missing values. Not here.

# In[192]:


features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>0]


# ### Step 2 - print the feature name and the percentage of missing values

# In[193]:


for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(),4), " % missing values")


# ### Since they are many missing values, we need to find the relationship between missing values and Sales Price
# 
# Let's plot some diagram for this relationship

# In[194]:


for feature in features_with_na:
    data=dataset.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(),1,0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# Here With the relation between the missing values and the dependent variable is clearly visible.So We need to replace these nan values with something meaningful which we will do in the Feature Engineering section
# 
# From the above dataset some of the features like Id are not required

# In[107]:


print("Number of Ids of Houses : {}".format(len(dataset.Id)))


# # Numerical Variables

# In[195]:


# list of numerical features
# O means object. Normal string field values
# if it's not object, by default it becomes numerical

numerical_features = [feature for feature in dataset.columns if dataset[feature].dtype !="O"]


# In[196]:


print( "Number of numerical features :", len(numerical_features))


# In[197]:


# visualise the numerical variables

dataset[numerical_features].head()


# ### Temporal Variables(Eg: Datetime Variables)
# 
# From the Dataset we have 4 year variables. We have extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold. We will be performing this analysis in the Feature Engineering which is the next video.

# In[198]:


# list of variables that contain year information

year_features = [feature for feature in numerical_features if "Yr" in feature or "Year" in feature]

year_features


# In[199]:


for feature in year_features:
    print(feature, dataset[feature].unique())


# ### Lets analyze the Temporal Datetime Variables
# ### We will check whether there is a relation between year the house is sold and the sales price
# 

# In[200]:


## we goupby by YrSold and take the media, for SalePrice

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price vs YearSold')


# In[114]:


year_features


# In[201]:


## Here we will compare the difference between All years features with SalePrice

for feature in year_features:
    if feature!="YrSold":
        data=dataset.copy()
        ## We will capture the difference between year variable and the year the house was sold for
        data[feature]=data["YrSold"]-data[feature]
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# ### Numerical variables are usually of 2 type
# 1. Continous variable 
# 2. Discrete Variables

# In[202]:


discrete_features=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_features+['Id']]


# In[203]:


print("Discrete Variables Count: {}".format(len(discrete_features)))


# In[204]:


discrete_features


# In[205]:


dataset[discrete_features].head()


# In[206]:


## Lets Find the relationship between them and Sale PRice

for feature in discrete_features:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# #### There is a relationship between variable number and SalePrice

# # Continuous Variables

# In[207]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_features+year_features+['Id']]


# In[208]:


print("Continuous features count: {}".format(len(continuous_features)))


# ### We need to find out the distribution of the continuous values!

# In[209]:


## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_features:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# ##### For some there is a Gaussian Distibution but not for others!
# 
# There are skewed data!
# 
# ### In the next part we will see how to perform a normalization
# 
# Whenever you are solving a regression problem, you should try to convert these none Gaussian distribution into Gaussian distribution

# # Exploratory Data Analysis Part 2

# ### We will be using logarithmic transformation

# ## We need to remove the outliers first
# 
# Performing a log transformation on a variable in machine learning does not necessarily remove outliers. Log transformation is a mathematical operation that can help to reduce the skewness of the data and make it more symmetrical. By taking the logarithm of the values, you can compress the range of larger values while expanding the range of smaller values.
# 
# However, outliers are extreme values that are far away from the majority of the data points and they can still exist after performing a log transformation. The effect of the log transformation on outliers depends on the specific distribution and characteristics of the data. In some cases, the transformation can moderate the impact of outliers by compressing the range of extreme values, but it does not remove them entirely.
# 
# To address outliers, additional preprocessing steps or outlier detection techniques can be employed. Some common approaches include:
# 
# Trimming or Winsorizing: Setting a threshold to cap or replace extreme values with a predetermined value.
# 
# Z-score or Standard Deviation: Identifying values that fall outside a certain number of standard deviations from the mean and considering them as outliers.
# 
# Interquartile Range (IQR): Identifying values that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR (where Q1 and Q3 are the first and third quartiles, respectively).
# 
# Robust methods: Utilizing statistical techniques that are less sensitive to outliers, such as robust regression or robust estimators.
# It's important to analyze the nature of your data and choose appropriate techniques to handle outliers effectively.

# In[124]:


## We will be using logarithmic transformation

## if we don't use copy and use just data=dataset, both refer to the same memory
## and any change to the data will affect dataset too

for feature in continuous_features:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()


# ### To find the outliers we use boxplot
# 
# 1. why did you take the log before detecting outliers? it's a practice I've never seen before.
# 
# 2. why didn't you apply Isolation Forest instead of the box plot, there are at least 2 upsides to it vs box plots, it will find outliers based on the relationship of multi-variables and by opting the right parameter it possible to get the index of outliers which is not possible in box plots.
# 
# 3. why not using the correlation matrix at least for continuous variables (I said it because somehow we can transform categorical variables to numerical by implementing kind of label encoding) since it is kind of high dimensional dataset, and in real projects, it's really time-consuming to explore scatter plots one-by-one.
# 
# 

# In[125]:


for feature in continuous_features:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        # data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# # Categorial Features

# In[126]:


categorial_features = [ feature for feature in dataset.columns if dataset[feature].dtype =="O"]


# In[127]:


categorial_features


# In[128]:


dataset[categorial_features].head(10)


# ### How many categories do we have inside our categorial features?

# In[129]:


for feature in categorial_features:
    print('The feature is {} and it has {} unique categories'.format(feature,(len(dataset[feature].unique()))))


# ### Find out the relationship between categorical variable and dependent feature SalesPrice

# In[130]:


for feature in categorial_features:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# # Feature Engineering

# ### We will be performing all the below steps in Feature Engineering
# 
# 1. Missing values
# 2. Temporal variables
# 3. Categorical variables: remove rare labels
# 4. Standardize the values of the variables to the same range

# In[131]:


# To visualise all the columns in the dataframe

pd.pandas.set_option('display.max_columns', None)


# In[132]:


dataset.head()


# ### Data Leakage 
# Always remember there may always be a chance of data leakage, so we need to split the data first and then apply feature Engineering.
# 
# Data leakage in machine learning refers to the unintentional or improper introduction of information from the training dataset into the model during the learning process, which can lead to overly optimistic performance estimates or biased predictions. It occurs when information that would not be available in a real-world scenario or at the time of prediction is used to train or evaluate the model, thereby compromising its generalization capability.
# 
# Data leakage can take several forms:
# 
# Training Data Leakage: This happens when information from the test set or future data is inadvertently included in the training process. For example, using future information as a predictor or including target variables that are only available after the prediction time.
# 
# Target Leakage: Target leakage occurs when the target variable (the variable to be predicted) is indirectly included in the feature set. This can happen when using information that is derived from the target variable or information that is only available after the target is determined. Target leakage leads to unrealistically high model performance during training but results in poor performance on new data.
# 
# Feature Leakage: Feature leakage occurs when information that would not be available during prediction is included as a feature in the model. For example, including time-dependent variables that would not be known at the time of prediction.
# 
# Data Preprocessing Leakage: Data preprocessing operations, such as scaling, normalization, or imputation, can introduce data leakage if they are performed using information from the entire dataset, including the test set. It's important to ensure that preprocessing steps are performed independently for the training and test sets.
# 
# To prevent data leakage, it is crucial to carefully partition the data into training and test sets before any preprocessing or model development. Additionally, understanding the domain and the temporal nature of the data can help identify potential sources of leakage and take appropriate measures to mitigate it. Regular validation and cross-validation techniques can also be employed to assess the model's performance on unseen data.

# In[133]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(dataset,dataset['SalePrice'],test_size=0.1,random_state=0)


# In[134]:


x_train.shape,x_test.shape


# In[135]:


y_train.shape,y_test.shape


# In[98]:


dataset.shape


# ### But this dataset is from kaggle and we already have a training and test data

# ## Feature Engineering for the train data
# 
# As a homework we do it for the test data too

# In[138]:


## Let us capture all the nan values
## First lets handle Categorical features which are missing

nan_features= [feature for feature in categorial_features if dataset[feature].isnull().sum()>0 and dataset[feature].dtype=='O']


# In[141]:


for feature in nan_features:
    print("{}: {}% missing values".format(feature, np.round(dataset[feature].isnull().mean(),4)))


# #### We create a new category for these NaN values

# In[142]:


def replace_cat_features(dataset, nan_features):
    data=dataset.copy()
    data[nan_features]=data[nan_features].fillna('Missing')
    return data


# In[143]:


dataset=replace_cat_features(dataset,nan_features)


# In[144]:


dataset[nan_features].isnull().sum()


# In[145]:


dataset.head()


# In[149]:


# Checking the missing values of the numerical features
numerical_with_nan=[feature for feature in numerical_features if dataset[feature].isnull().sum()>1 and dataset[feature].dtype!='O']


# In[150]:


for feature in numerical_with_nan:
    print("{}: {}% NaN values".format(feature,np.round(dataset[feature].isnull().mean(),4)))


# #### We replace the NaN values of the numerical features with the median, since there are outliers

# In[152]:


# Replacing the numerical Missing Values

for feature in numerical_with_nan:
    ## Since There are outlier, we will replace by using median 
    median_value = dataset[feature].median()
    
    ## create a a new feature to capture NaN values
    dataset[feature+"NaN"] = np.where(dataset[feature].isnull(),1,0)
    
    dataset[feature].fillna(median_value,inplace=True)


# In[153]:


dataset[numerical_with_nan].isnull().sum()


# In[155]:


dataset.head(10)


# In[157]:


year_features


# ## Temporal Variables
# 
# year_features contains my temporal variables.
# 
# We found out that with year sold, the price was actually reducing as it increased.And it's not normal.
# 
# So instead of just focusing on Year Sold, we try to find the relationship between other temporal features and year sold.

# In[158]:


## Temporal variables (Date Time Variables)

for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    dataset[feature]=dataset['YrSold']-dataset[feature]


# In[160]:


dataset.head(10)


# In[161]:


dataset[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']]


# #### Log Normal Distribution
# 
# Sine the numerical variables are skewed we will perform log normal distribution.
# 
# Skewed means that it is not in the form of Gaussian Normal Distribution.

# In[162]:


dataset.head()


# Many of the numerical features have skewed values.
# 
# Consider that those numerical features must not have zero values inside them.
# 
# If it has zero values just skip it!
# 
# We saw that these features have skewed values:
# 'LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice'

# In[163]:


## performing log normal distribution

num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    dataset[feature]=np.log(dataset[feature])


# In[164]:


dataset.head()


# ### Handling Rare Categorical Feature
# We will remove categorical variables that are present in less than 1% of the observations

# In[167]:


categorial_features = [feature for feature in dataset.columns if dataset[feature].dtype=='O']


# In[168]:


categorial_features


# In[169]:


len(dataset)


# In[171]:


# we groupby the categories that are inside the categorial features

for feature in categorial_features:
    # calculating the precentage of each category with respect to the hole dataset
    temp = dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    
    # if the amount is greater than 0.01, we take the index of that amount
    temp_df = temp[temp>0.01].index
    dataset[feature] = np.where(dataset[feature].isin(temp_df),dataset[feature],"Rare_Value")


# In[172]:


dataset.head()


# ## Here, Log normalisation will handle outliers. 
# 
# 
# In machine learning, the treatment of outliers depends on the specific problem at hand and the characteristics of the data. Here are a few common approaches to handling outliers:
# 
#     - Removing outliers: One option is to remove outliers from the dataset. This can be appropriate when the outliers are likely to be due to errors or noise in the data. However, it's important to be cautious when removing outliers, as they may contain valuable information or represent legitimate extreme values. Removing outliers should be done carefully and based on domain knowledge or statistical techniques.
# 
#     - Transforming the data: Another approach is to transform the data to make it more resilient to outliers. For example, you can apply mathematical transformations such as logarithmic or power transformations to reduce the impact of extreme values. These transformations can help normalize the data and make it more suitable for certain machine learning algorithms.
# 
#     - Using robust models: Some machine learning algorithms are inherently robust to outliers. For instance, tree-based models like Random Forests or Gradient Boosting Machines can handle outliers to some extent, as they partition the data into regions. Additionally, robust regression techniques like RANSAC (RANdom SAmple Consensus) or Theil-Sen estimator can be used to estimate model parameters, which are less influenced by outliers.
# 
#     - Binning or discretizing: In certain cases, you may choose to bin or discretize continuous variables. By converting numerical values into categories, you can diminish the effect of outliers. This approach is useful when you believe that the outlier values are erroneous or when the relationship between the variable and the target is nonlinear.
# 
#     - Treating outliers as a separate class: In some scenarios, outliers may be of particular interest. Instead of removing them, you can label outliers as a distinct class and train a classification model to identify and classify them. This approach is common in anomaly detection tasks or when outliers represent rare events or critical instances.
# 
# It's important to note that the choice of approach depends on the context, the nature of the data, and the specific machine learning problem. It's recommended to analyze and understand the outliers before deciding on an appropriate strategy.

# In[176]:


for feature in categorial_features:
    ## we take the mean value of the dependent feature for each category and sort them
    ## then add the categories and their index to a dictionary
    labels_ordered = dataset.groupby(feature)['SalePrice'].mean().sort_values().index
    labels_ordered = {k:i for i,k in enumerate(labels_ordered,0)}
    
    ## maping the categories to their index and transforming the data
    dataset[feature] = dataset[feature].map(labels_ordered)


# In[177]:


dataset.head()


# ##### map()
# Here, the map() function is applied to the selected column or feature. The map() function is a pandas method used to transform the values of a Series based on a mapping or function.
# 
# Mapfor example  here the map function asigns the values of labels_ordered which are {'Abnorml': 0, 'Rare_Value': 1, 'Family': 2, 'Normal': 3, 'Partial': 4} to the SaleCondition columns
# 

# ##### enumerate()
# 
# In Python, the enumerate() function is a built-in function that allows you to iterate over a sequence while also keeping track of the index of each item in the sequence. It adds a counter to an iterable and returns an enumerate object, which contains pairs of index and value for each element in the iterable.
# 
# The enumerate() function is particularly useful when you need to access both the index and value of each element in a sequence simultaneously. It eliminates the need to manually manage an index variable and provides a convenient way to iterate over items with their corresponding positions.
# 

# ## Feature Scaling
# 
# The final step is feature scaling
# 
# We do not perform it on Id and SalePrice columns because we drop Id, and SalePrice is also our dependent feature
# 
# Standard Scaler also works well
# 
# Actually the guy tried both approaches and got better results with MinMaxScaler
# 
# Standard Scaler : Standard Normal Distribution. Between -1 and 1
# 
# MinMAx Scaler : Between 0 and 1
# 

# In[179]:


feature_scale=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]


# In[180]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(dataset[feature_scale])


# In[181]:


# transform the train and test set, and add on the Id and SalePrice variables
# we first take out the 'Id' and 'SalePrice', then transform it

x_train = pd.concat([dataset[['Id','SalePrice']].reset_index(drop=True),
                   pd.DataFrame(scaler.transform(dataset[feature_scale]),columns=feature_scale)],
                    axis=1)


# scaler . transform (dataset [feature_scale] )
# 
# returns an array and this array needs to be converted to a dataframe

# In[182]:


x_train.head()


# ###### Now the data is scaled and ready to be put in the machine learning algorithm

# In[186]:


dataset.to_csv('House Prices/X_train.csv',index=False)

# because we do not want any default indexes


#  ### The same thing should be done for the Test Data

# In[211]:


dataset=pd.read_csv('House Prices/test.csv')


# In[212]:


dataset.head()


# In[213]:


## print shape of the dataset with rows and columns
print(dataset.shape)


# In Data Analysis We will Analyze To Find out the below stuff
# 
#     1. Missing Values
#     2. All The Numerical Variables
#     3. Distribution of the Numerical Variables
#     4. Categorical Variables
#     5. Cardinality of Categorical Variables
#     6. Outliers
#     7. Relationship between independent and dependent feature(SalePrice)

# The cardinality of a categorical variable refers to the number of unique categories or distinct values that the variable can take. It represents the size or the count of the unique levels within the categorical variable.

# # 1. Missing Values
# 
# ### Here we will check the percentage of nan values present in each feature
# ### Step 1 - make the list of features which has missing values
# 
# In feature engineering we will handle the missing values. Not here.

# In[215]:


features_with_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>0]


# In[216]:


features_with_nan


# ### Step 2 - print the feature name and the percentage of missing values

# In[218]:


for feature in features_with_nan:
    print("{}: {}% Missing values".format(feature, np.round(dataset[feature].isnull().mean(),4)))


# ### Since they are many missing values, we need to find the relationship between missing values and Sales Price. But this is our test set and we dont have SalePrice

# From the above dataset some of the features like Id are not required

# In[221]:


print("Number of Ids of Houses : {}".format(len(dataset.Id)))


# # Numerical Variables

# In[222]:


# list of numerical features
# O means object. Normal string field values
# if it's not object, by default it becomes numerical

numerical_features = [feature for feature in dataset.columns if dataset[feature].dtype !="O"]


# In[223]:


print( "Number of numerical features :", len(numerical_features))


# In[224]:


# visualise the numerical variables

dataset[numerical_features].head()


# ### Temporal Variables(Eg: Datetime Variables)
# 
# From the Dataset we have 4 year variables. We have extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold. We will be performing this analysis in the Feature Engineering.

# In[225]:


# list of variables that contain year information

year_features = [feature for feature in numerical_features if "Yr" in feature or "Year" in feature]

year_features


# In[226]:


for feature in year_features:
    print(feature, dataset[feature].unique())


# ### Numerical variables are usually of 2 type
# 1. Continous variable 
# 2. Discrete Variables

# In[227]:


discrete_features=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_features+['Id']]


# In[228]:


print("Discrete Variables Count: {}".format(len(discrete_features)))


# In[229]:


discrete_features


# In[230]:


dataset[discrete_features].head()


# # Continuous Variables

# In[231]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_features+year_features+['Id']]


# In[232]:


print("Continuous features count: {}".format(len(continuous_features)))


# In[233]:


## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_features:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# ##### For some there is a Gaussian Distibution but not for others!
# 
# There are skewed data!
# 
# ### In the next part we will see how to perform a normalization
# 
# Whenever you are solving a regression problem, you should try to convert these none Gaussian distribution into Gaussian distribution

# ### Here, to find the outliers we use boxplot

# In[235]:


for feature in continuous_features:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        # data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# # Categorail Features

# In[236]:


categorial_features = [ feature for feature in dataset.columns if dataset[feature].dtype =="O"]


# In[237]:


categorial_features


# In[238]:


dataset[categorial_features].head(10)


# ### How many categories do we have inside our categorial features?

# In[239]:


for feature in categorial_features:
    print('The feature is {} and it has {} unique categories'.format(feature,(len(dataset[feature].unique()))))


# # Feature Engineering

# ### We will be performing all the below steps in Feature Engineering
# 
# 1. Missing values
# 2. Temporal variables
# 3. Categorical variables: remove rare labels
# 4. Standardize the values of the variables to the same range

# In[240]:


# To visualise all the columns in the dataframe

pd.pandas.set_option('display.max_columns', None)


# In[241]:


dataset.head()


# ## Let us capture all the nan values
# ## First lets handle Categorical features which are missing

# In[242]:


nan_features= [feature for feature in categorial_features if dataset[feature].isnull().sum()>0 and dataset[feature].dtype=='O']


# In[243]:


for feature in nan_features:
    print("{}: {}% missing values".format(feature, np.round(dataset[feature].isnull().mean(),4)))


# #### We create a new category for these NaN values

# In[244]:


def replace_cat_features(dataset, nan_features):
    data=dataset.copy()
    data[nan_features]=data[nan_features].fillna('Missing')
    return data


# In[245]:


dataset=replace_cat_features(dataset,nan_features)


# In[246]:


dataset[nan_features].isnull().sum()


# In[247]:


dataset.head()


# In[248]:


# Checking the missing values of the numerical features
numerical_with_nan=[feature for feature in numerical_features if dataset[feature].isnull().sum()>0 and dataset[feature].dtype!='O']


# In[249]:


for feature in numerical_with_nan:
    print("{}: {}% NaN values".format(feature,np.round(dataset[feature].isnull().mean(),4)))


# #### We replace the NaN values of the numerical features with the median, since there are outliers

# In[250]:


# Replacing the numerical Missing Values

for feature in numerical_with_nan:
    ## Since There are outlier, we will replace by using median 
    median_value = dataset[feature].median()
    
    ## create a a new feature to capture NaN values
    dataset[feature+"NaN"] = np.where(dataset[feature].isnull(),1,0)
    
    dataset[feature].fillna(median_value,inplace=True)


# In[251]:


dataset[numerical_with_nan].isnull().sum()


# In[252]:


dataset.head(10)


# ## Temporal Variables
# 
# year_features contains my temporal variables.
# 
# We found out that with year sold, the price was actually reducing as it increased.And it's not normal.
# 
# So instead of just focusing on Year Sold, we try to find the relationship between other temporal features and year sold.

# In[254]:


## Temporal variables (Date Time Variables)

for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    dataset[feature]=dataset['YrSold']-dataset[feature]


# In[255]:


dataset[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']]


# #### Log Normal Distribution
# 
# Sine the numerical variables are skewed we will perform log normal distribution.
# 
# Skewed means that it is not in the form of Gaussian Normal Distribution.

# Many of the numerical features have skewed values.
# 
# Consider that those numerical features must not have zero values inside them.
# 
# If it has zero values just skip it!
# 
# We saw that these features have skewed values:
# 'LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea'

# In[258]:


## performing log normal distribution

num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

for feature in num_features:
    dataset[feature]=np.log(dataset[feature])


# In[259]:


dataset.head()


# ### Handling Rare Categorical Feature
# We will remove categorical variables that are present in less than 1% of the observations

# In[260]:


categorial_features = [feature for feature in dataset.columns if dataset[feature].dtype=='O']
categorial_features 


# In[261]:


len(dataset)


# In[263]:


# we groupby the categories that are inside the categorial features

for feature in categorial_features:
    # calculating the precentage of each category with respect to the hole dataset
    temp = dataset.groupby(feature)['Id'].count()/len(dataset)
    
    # if the amount is greater than 0.01, we take the index of that amount
    temp_df = temp[temp>0.01].index
    dataset[feature] = np.where(dataset[feature].isin(temp_df),dataset[feature],"Rare_Value")


# In[264]:


dataset.head()


# #### Here, Log normalisation will handle outliers. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




