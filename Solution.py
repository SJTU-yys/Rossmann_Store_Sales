# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:23:14 2015

@author: yys
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from sklearn.cross_validation import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import os

def create_feature_map(features):
    outfile = open('predicted/xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

def build_features(features, data):
    data.fillna(0,inplace = True)
    data.loc[data.Open.isnull(),'Open'] = 1
    features.extend(['Store', 'CompetitionDistance', 'Promo',  'Promo2', 'SchoolHoliday'])
    
    # encode factor features
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {"0":0,'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)
    
    # Convert the date
    features.extend(['DayOfWeek','Month','Day','Year','WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['WeekOfYear'] = data.Date.dt.weekofyear
    
    # Calculate the time interval since the competition open, promotion open
    features.extend(['CompetitionOpen', 'PromoOpen'])
    data['CompetitionOpen'] = 12*(data['Year']-data['CompetitionOpenSinceYear']) + \
                            data['Month']-data['CompetitionOpenSinceMonth']

    data['PromoOpen'] = 12*(data['Year']-data['Promo2SinceYear']) + \
                            (data['Month']-data['Promo2SinceWeek'])/4.0    
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0
    
    # Judge whether the month is in promotion
    features.extend(['IsPromoMonth'])
    data['IsPromoMonth'] = 0
    Month2Str = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    data['monthStr'] = data.Month.map(Month2Str)    
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    for months in data.PromoInterval.unique():
        if months != '':
            for m in months.split(','):
                data.loc[(data.monthStr == m) & (data.PromoInterval == months), 'IsPromoMonth'] = 1
                
    # We get the average customers of a store
    
    return data
    

types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}

os.chdir('/Users/yys/Documents/Kaggle/Rossmann_Store_Sales')
training_data = pd.read_csv('train.csv', parse_dates = [2], dtype=types)
store = pd.read_csv('store.csv')
#test_data = pd.read_csv('test.csv', parse_dates=[3], dtype=types)
training_data = training_data[training_data["Open"] != 0]
training_data = training_data[training_data["Sales"] > 0]

training_data = pd.merge(training_data, store, on = 'Store')
features = []

build_features(features, training_data)

## 1 Get the average number of customers of a store
store_group = training_data.groupby(training_data.Store, as_index= False)
ave_customers = store_group.Customers.aggregate(np.mean)
training_data = pd.merge(training_data, ave_customers, on='Store')
training_data.rename(columns={'Customers_y': 'MeanCustomers', 'Customers_x':'Customers'}, inplace=True)

test = pd.merge(test, ave_customers, on='Store')
test.rename(columns={'Customers': 'MeanCustomers'}, inplace=True)

## 2 mean(Sales)
mean_sales = store_group.Sales.aggregate(np.mean)
#training_data['SalePerCustomer'] = mean_sales/training_data['ave_customers']
training_data = pd.merge(training_data, mean_sales, on='Store')
training_data.rename(columns={'Sales_y': 'MeanSales', 'Sales_x':'Sales'}, inplace=True)
training_data.MeanSales = np.log(training_data.MeanSales)

test = pd.merge(test, mean_sales, on='Store')
test.rename(columns={'Sales': 'MeanSales'}, inplace=True)
test.MeanSales = np.log(test.MeanSales)

## 3 mean(Sales)/mean(Customers)
training_data['SalePerCustomer'] = training_data.MeanSales.div(training_data.MeanCustomers)

test['SalePerCustomer'] = test.MeanSales.div(test.MeanCustomers)

## 4 mean(Sales) per day-of-week
store_day_of_week_group = training_data.groupby([training_data.Store, training_data.DayOfWeek], as_index = False)
mean_sales_per_weekday = store_day_of_week_group.mean()[['Sales','Store','DayOfWeek','Customers']]
training_data = pd.merge(training_data, mean_sales_per_weekday[['Store','DayOfWeek','Sales']], on=['Store','DayOfWeek'])
training_data.rename(columns={'Sales_y': 'MeanSalesPerDayOfWeek', 'Sales_x':'Sales'}, inplace=True)
training_data.MeanSalesPerDayOfWeek = np.log(training_data.MeanSalesPerDayOfWeek)

test = pd.merge(test, mean_sales_per_weekday[['Store','DayOfWeek','Sales']], how = 'left', on=['Store','DayOfWeek'])
test.rename(columns={'Sales': 'MeanSalesPerDayOfWeek'}, inplace=True)
test.MeanSalesPerDayOfWeek = np.log(test.MeanSalesPerDayOfWeek)
# get the unique rows of training_data
#training_data.loc[~training_data[['Store','DayOfWeek']].duplicated()

## 5 mean(Sales)/mean(Customers) per weekday
mean_sales_per_weekday['SalesPerCustomerWeekday'] = mean_sales_per_weekday['Sales'].div(mean_sales_per_weekday['Customers'])
training_data = pd.merge(training_data, mean_sales_per_weekday[['Store','DayOfWeek','SalesPerCustomerWeekday']], on=['Store','DayOfWeek'])
#training_data.rename(columns={'SalePerCustomerWeekday': 'SalesPerCustomerWeekday'}, inplace=True)

test = pd.merge(test, mean_sales_per_weekday[['Store','DayOfWeek','SalesPerCustomerWeekday']], how='left', on=['Store','DayOfWeek'])

## 6 mean(Sales)/ promo
store_promotion_group = training_data.groupby([training_data.Store, training_data.Promo], as_index = False)
mean_sales_per_store_promo = store_promotion_group.mean()[['Sales','Store','Promo']]
training_data = pd.merge(training_data, mean_sales_per_store_promo, on=['Store','Promo'])
training_data.rename(columns={'Sales_y': 'MeanSalesPerStorePromo', 'Sales_x':'Sales'}, inplace=True)
training_data.MeanSalesPerStorePromo = np.log(training_data.MeanSalesPerStorePromo)

test = pd.merge(test, mean_sales_per_store_promo, how='left', on=['Store','Promo'])
test.rename(columns={'Sales': 'MeanSalesPerStorePromo'}, inplace=True)
test.MeanSalesPerStorePromo = np.log(test.MeanSalesPerStorePromo)

## 7 customer type according to store type
training_data['CustomerType'] = training_data['StoreType']
test['CustomerType'] = test['StoreType']

## 8 mean(customer) and mean(sales) per Storetype and assortment
store_type_assortment_group = training_data.groupby([training_data.StoreType, training_data.Assortment], as_index = False)
mean_customer_per_store_type_assortment = store_type_assortment_group.mean()[['Sales','Customers','StoreType','Assortment']]
training_data = pd.merge(training_data, mean_customer_per_store_type_assortment, on = ['StoreType','Assortment'])
training_data.rename(columns={'Sales_y':'SalessPerStoreTypeAssortment','Sales_x':'Sales','Customers_y': 'CustomersPerStoreTypeAssortment', 'Customers_x':'Customers'}, inplace=True)
training_data.SalessPerStoreTypeAssortment = np.log(training_data.SalessPerStoreTypeAssortment)

test = pd.merge(test, mean_customer_per_store_type_assortment, how = 'left', on = ['StoreType','Assortment'])
test.rename(columns={'Sales':'SalessPerStoreTypeAssortment','Customers': 'CustomersPerStoreTypeAssortment'}, inplace=True)
test.SalessPerStoreTypeAssortment = np.log(test.SalessPerStoreTypeAssortment)

## 9 divide the combination of storetype and assortment into 4 group
training_data['SalesType'] = 1
training_data.loc[(training_data.StoreType == 1) & (training_data.Assortment == 2), 'SalesType'] = 2
training_data.loc[(training_data.StoreType == 2) & (training_data.Assortment == 2), 'SalesType'] = 3
training_data.loc[(training_data.StoreType == 3) & (training_data.Assortment == 2), 'SalesType'] = 4

test['SalesType'] = 1
test.loc[(test.StoreType == 1) & (test.Assortment == 2), 'SalesType'] = 2
test.loc[(test.StoreType == 2) & (test.Assortment == 2), 'SalesType'] = 3
test.loc[(test.StoreType == 3) & (test.Assortment == 2), 'SalesType'] = 4

training_data.to_csv('training_data', ';', index=False)
features.extend(['MeanCustomers','MeanSales','SalePerCustomer','MeanSalesPerDayOfWeek','SalesPerCustomerWeekday','MeanSalesPerStorePromo','SalessPerStoreTypeAssortment','CustomersPerStoreTypeAssortment','CustomerType','SalesType'])

### Feature Engineering end
print "Start Training"
X_train,X_valid = train_test_split(training_data, test_size=0.012, random_state = 10)
cols = np.array(training_data.columns)
#cols = [u'Store', u'DayOfWeek', u'Date', u'Sales', u'Customers', u'Open', u'Promo', u'StateHoliday', \
#u'SchoolHoliday', u'StoreType', u'Assortment', u'CompetitionDistance', u'CompetitionOpenSinceMonth', \
#u'CompetitionOpenSinceYear', u'Promo2', u'Promo2SinceWeek', u'Promo2SinceYear', u'PromoInterval', u'Year', \
#u'Month', u'Day', u'WeekOfYear', u'CompetitionOpen', u'PromoOpen', u'IsPromoMonth', u'monthStr']

tmp = pd.DataFrame(X_train, columns=cols)
X_train = tmp
tmp = pd.DataFrame(X_valid, columns=cols)
X_valid = tmp

y_train = np.log1p(np.array(X_train.Sales, dtype=int))
tmp = pd.DataFrame(y_train, columns = ['Sales'], dtype='float')
y_train = tmp
y_valid = np.log1p(np.array(X_valid.Sales, dtype=int))
tmp = pd.DataFrame(y_valid, columns = ['Sales'], dtype='float')
y_valid = tmp

# Build the DMatrix for xgboost
a = X_train[features]
for c in a.columns:
    a[c] = a[c].astype(float)
b = X_valid[features]
for c in a.columns:
    b[c] = b[c].astype(float)
dtrain = xgb.DMatrix(a, y_train)
dvalid = xgb.DMatrix(b, y_valid)

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 500
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

#==============================================================================
test = pd.read_csv("test.csv", parse_dates=[3], dtype=types)
test.fillna(1, inplace=True)
test = pd.merge(test, store, on='Store')
build_features([], test)
test.fillna(0, inplace=True)
dtest = xgb.DMatrix(test[features])

## Store the model file
print "Storing the model file..."
import pickle as pk
model_file = open('predicted/model_new.pkl','w')
pk.dump(gbm, model_file)
model_file.close()

print "Predicting on test data set..."
test_probs = gbm.predict(dtest)
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})

# If the store is closed the Sales should be zero
result.loc[test.Open==0, 'Sales'] = 0
result.to_csv("predicted/xgboost_8_submission.csv", index=False)

# XGB feature importances
# Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code
create_feature_map(features)
importance = gbm.get_fscore(fmap='predicted/xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
