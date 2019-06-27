'''
This is the data pre-processing template which can be used to process the data before using any model
'''

# Importing the libaries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the data set

data = pd.read_csv('path/to/dataset.csv')
X = data.iloc[:,:-1].values # HERE -1 ARE DUMMY VALUES
y = data.iloc[:,-1].values # HERE -1 ARE DUMMY VALUES

# Dealing with Missing data

# 1) remove that row
# 2) taking the mean of the column and replacing the missing data with the mean

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', stategy='mean', axis=0)
imputer.fit(X[:, 1:3]) # here pass the column index e.g. 1,2
X[:, 1:3] = imputer.transform(X[:, 1:3]) # replaces the missing data in X

# Encoding the categorical variables

X_new = pd.get_dummies(X)

# Dependent variable we need to use only LabelEncoder (NOT ALWAYS NEEDED CHECK IF YOU REALLY SEE THE NEED TO DO THIS. GOOGLE ABOUT WHEN YOU SHOULD DO THIS...)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Spliting data into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train  = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)

# Get the model from sklearn like LinearRegressor, SVM, e.t.c
	
	# Try google for this. Get the model and pass the required components. Then fit your model and then predict the output.

# Calculate the score for predicted y values using scorers like f1, f-beta, R^2 score

	# check sklearn.metrics for the scorers.
