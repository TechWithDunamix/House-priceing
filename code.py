import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
file_name = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)

print(df.dtypes)
df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)
print(df.describe())
# Check missing values
print("Number of NaN values for the column bedrooms:", df['bedrooms'].isnull().sum())
print("Number of NaN values for the column bathrooms:", df['bathrooms'].isnull().sum())

# Replace missing values with the mean
mean_bedrooms = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean_bedrooms, inplace=True)

mean_bathrooms = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan, mean_bathrooms, inplace=True)

# Verify that the missing values have been handled
print("Number of NaN values for the column bedrooms:", df['bedrooms'].isnull().sum())
print("Number of NaN values for the column bathrooms:", df['bathrooms'].isnull().sum())
floors_df = df['floors'].value_counts().to_frame()
print(floors_df)
sns.boxplot(x='waterfront', y='price', data=df)
plt.show()
sns.regplot(x='sqft_above', y='price', data=df)
plt.show()
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
print("R^2 for `long` feature:", lm.score(X, Y))

X = df[['sqft_living']]
Y = df['price']
lm.fit(X, Y)
print("R^2 for `sqft_living` feature:", lm.score(X, Y))
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features]
lm.fit(X, Y)
print("R^2 for multiple features:", lm.score(X, Y))
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(X, Y)
print("R^2 for polynomial features:", pipe.score(X, Y))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
print("Number of test samples:", X_test.shape[0])
print("Number of training samples:", X_train.shape[0])
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, Y_train)
print("R^2 for Ridge regression:", ridge_model.score(X_test, Y_test))
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

ridge_model_poly = Ridge(alpha=0.1)
ridge_model_poly.fit(X_train_poly, Y_train)
print("R^2 for polynomial Ridge regression:", ridge_model_poly.score(X_test_poly, Y_test))

