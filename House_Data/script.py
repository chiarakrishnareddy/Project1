import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import geopandas as gpd
from matplotlib.colors import Normalize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
import lib as lb


pd.options.display.float_format = '{:.10f}'.format
df_housing = pd.read_csv('~/desktop/House_Data/USA_Housing.csv')
#check how many null values are in df


df_housing.dropna(inplace=True)

# drop out fields that aren't useful 
df_housing.drop(columns= ["street","floors","waterfront", "state", "yr_renovated", "country", "date", "city", "yr_built", "zip", "sqft_lot", "bathrooms", "condition"], inplace = True)

# step 2: coeff matrix to determine feature relationships
plt.figure(figsize=(10,10))
correlation_matrix = df_housing.corr()
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm")
#plt.show()

# step 3: train test split
Y = df_housing["price"]
X = df_housing.drop(columns=["price"])
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, random_state = 8)
lb.linearReg(xtrain,ytrain,xtest,ytest)
lb.decisionTree(xtrain, ytrain, xtest, ytest)
lb.randomForest(xtrain, ytrain, xtest, ytest)
