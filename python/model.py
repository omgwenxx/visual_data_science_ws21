import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor


df = pd.read_csv("data/final/final_merge_region.csv")
# remove countries that are missing score
df = df[~df['score'].isnull()]

# train data 2015-2020
past_data = df.loc[df.year < 2021]
X = past_data.loc[:, past_data.columns != "score"]
X = X.drop(["year"],axis=1)
y = past_data["score"]

# test data 2021
future_data = df.loc[df.year > 2020]
X_test = future_data.loc[:, future_data.columns != "score"]
X_test = X_test.drop(["year"],axis=1)
y_test = future_data["score"]

# preprocessing dataset in order to work for linear regression
# remove columns that are missing more than 80% of their values
col_keep = X.columns[X.isnull().mean() < 0.8] # col with less than 80% missing
X = X[col_keep]
X_test = X_test[col_keep]

# remove highly correlated features to avoid colinearity
cor_matrix = X.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X = X.drop(to_drop, axis=1)
X_test = X_test.drop(to_drop, axis=1)

# variance threshold
vt = VarianceThreshold()
geo_col = ["country","sub-region","region"]
vt.fit(X.loc[:, X.columns.difference(geo_col)])
X_diff = X.loc[:, X.columns.difference(geo_col)]
X_var = X_diff.iloc[:, vt.variances_ > 0.3]
#X = pd.concat([X_var,X.loc[:,geo_col]],axis=1) # concat along columns
X = X_var
# one-hot-enconding
#X = pd.get_dummies(X, columns=geo_col)

# impute missing values
imputer = KNNImputer()
imputer.fit(X)
X_ = imputer.transform(X)
X[X.columns] = X_
# fit model

regsnames = [(AdaBoostRegressor().fit(X, y), "ada"),
             (RandomForestRegressor().fit(X, y), "rf"),
             (LinearRegression().fit(X, y), "lr")]

# predict new values
X_diff_test = X_test.loc[:, X_test.columns.difference(geo_col)]
X_var_test = X_diff_test.iloc[:, vt.variances_ > 0.3]
#X_test = pd.concat([X_var_test, X_test.loc[:, geo_col]], axis=1)
X_test = X_var_test
#X_test = pd.get_dummies(X_test, columns=geo_col)
missing_col = X.columns.difference(X_test.columns)
X_test[missing_col] = 0
X_test_ = imputer.transform(X_test)
X_test[X_test.columns] = X_test_

for (reg, name) in regsnames:

    pred = reg.predict(X_test)

    results = pd.DataFrame({"y_true":y_test,"y_pred":pred})
    results.to_csv(f"./data/model_results_{name}_nocountries_withvariancestuff.csv")

    print(f"MSE:({round(mse(y_test,pred),2)})")
    print(f"R^2:({round(r2_score(y_test,pred),2)})")


    print(((y_test - y.mean())**2).mean())

