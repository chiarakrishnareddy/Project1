import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import geopandas as gpd
from matplotlib.colors import Normalize
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression



def show_barplt(df_housing):
    bins = [0,500,1000,1500,2000,2500,3000,3500,4000, df_housing["sqft_living"].max()]
    labels = ["<500", "500 - 1000", "1000 - 1500", "1500 - 2000", "2000 - 2500", "2500 - 3000", "3000 - 3500", "3500 - 4000", ">4000"]
    df_housing["sqft_bins"] = pd.cut(df_housing["sqft_living"], bins = bins, labels = labels)
    sns.set(style= "whitegrid", )
    plt.figure(figsize=(10,10))
    barOne = sns.barplot(x= "sqft_bins", y= "price", data= df_housing, hue = "sqft_bins", legend = False, palette= 'viridis')
    plt.title = "avg house price per bedroom"
    plt.xlabel = "living space"
    plt.ylabel = "price"
    barOne.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'${x:,.0f}'))
    plt.show()

'''
def show_statemap(df_housing):
    fig,ax = plt.subplots(figsize = (10,10))
    #world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    #usa = world[world.name == "United States of America"].copy()
    usa = gpd.read_file('~/desktop/House_Data/cb_2018_us_state_500k')
    zip_code = gpd.read_file('~/desktop/House_Data/housing_zip_codes')
    merge = usa.merge(df_housing, left_on = "STUSPS", right_on = "state")
    print(type(merge))
    merge.plot(ax=ax, edgecolor = 'black', color = 'blue', linewidth = 1.0, )
    zip_code['ZCTA5CE10'] = zip_code['ZCTA5CE10'].apply(lambda x: int(x))
    print(type(zip_code))
    merge_zip = zip_code.merge(merge, left_on = 'ZCTA5CE10', right_on = 'zip')
    geo_merge_zip = gpd.GeoDataFrame(merge_zip)
    geo_merge_zip.set_geometry('geometry_x', inplace = True)
    geo_merge_zip["price"] = geo_merge_zip['price'].apply(lambda x: x/1000)
    print(type(geo_merge_zip))
    norm = Normalize(vmin = geo_merge_zip['price'].min(), vmax = geo_merge_zip['price'].quantile(0.95))
    geo_merge_zip.plot(ax = ax, edgecolor = 'black', linewidth = 1.0, column = 'price', cmap = 'Reds', legend = True, norm = norm)

    print(geo_merge_zip.columns)

'''


def linearReg(xtrain,ytrain,xtest,ytest):
    # step 4: fitting the model to the data 
    model = LinearRegression()
    model.fit(xtrain, ytrain) 

    # step 5: make predicitions based on model 
    pred = model.predict(xtest)
    print(pred)

    # step 6: 
    correlation = np.corrcoef(ytest, pred)
    print(correlation[0][1])
    print(xtest.columns)
    print(model.coef_)
    allcoeff = pd.DataFrame({
        'feature': xtest.columns, 
        'coefficent': model.coef_,
    })
    allcoeffsorted = allcoeff.reindex(allcoeff["coefficent"].abs().sort_values(ascending = False).index)
    MSE = mean_squared_error(ytest, pred) ** 0.5
    print(f"MSE LinearReg {MSE:.2f}")


    # step 7: graph predications and actuals
    #plotting(ytest, pred)
    #plt.show()

def decisionTree(xtrain, ytrain, xtest, ytest):
    '''
    this type of model works well on non-linear data 
    '''
    model = DecisionTreeRegressor(random_state = 10)
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    correlation = np.corrcoef(ytest, pred)[0][1]
    MSE = mean_squared_error(ytest, pred) ** 0.5
    print(f"MSE DT {MSE:.2f}")

    #plotting(ytest, pred)


def plotting(ytest, pred):
    plt.figure(figsize = (10,10))
    plt.scatter(ytest, pred, color='blue', label = 'actual Vs pred')
    plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color = 'red', linewidth = 10, label = 'perfect pred')
    plt.xlabel('actual')
    plt.ylabel('pred')
    #plt.show()

def randomForest(xtrain, ytrain, xtest, ytest):
    '''
    this type of model works well on non-linear data 
    random forest is a collection of multiple DT which is the n estimator
    '''
    model = RandomForestRegressor(n_estimators = 100, random_state = 10)
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    correlation = np.corrcoef(ytest, pred)[0][1]
    MSE = mean_squared_error(ytest, pred) ** 0.5
    print(f"MSE RF {MSE:.2f}")

def supportVector(xtrain, ytrain, )
    #plotting(ytest, pred)