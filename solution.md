Alright, let's go through the steps manually. Here's how you can approach each part of the assignment based on the tasks you shared earlier. I'll walk you through each question and the solution process.

### Step 1: Importing Data and Libraries
First, ensure you have all the necessary libraries imported. These libraries are essential for data manipulation, visualization, and model development.

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline
```

Next, load the dataset:

```python
file_name = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)
```

### Step 2: Data Exploration
Use the `head()` function to display the first five rows of the dataframe:

```python
df.head()
```

#### Question 1: Display Data Types
To display the data types of each column, use the `dtypes` attribute:

```python
print(df.dtypes)
```

This will show you what kind of data each column holds, such as integers, floats, or objects (strings).

### Step 3: Data Wrangling

#### Question 2: Drop Unnecessary Columns
Drop the `id` and `Unnamed: 0` columns:

```python
df.drop(columns=['id', 'Unnamed: 0'], inplace=True)
df.describe()
```

This removes unnecessary columns that don't contribute to the analysis.

#### Handle Missing Values
Check for missing values in the `bedrooms` and `bathrooms` columns:

```python
print("number of NaN values for the column bedrooms:", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms:", df['bathrooms'].isnull().sum())
```

Replace the missing values with the mean:

```python
mean_bedrooms = df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean_bedrooms, inplace=True)

mean_bathrooms = df['bathrooms'].mean()
df['bathrooms'].replace(np.nan, mean_bathrooms, inplace=True)

print("number of NaN values for the column bedrooms:", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms:", df['bathrooms'].isnull().sum())
```

### Step 4: Exploratory Data Analysis

#### Question 3: Count Houses by Number of Floors
Use `value_counts()` to count how many houses have unique floor values:

```python
floors_count = df['floors'].value_counts().to_frame()
print(floors_count)
```

#### Question 4: Boxplot for Waterfront vs. Non-Waterfront Properties
Create a boxplot to compare house prices for waterfront and non-waterfront properties:

```python
sns.boxplot(x='waterfront', y='price', data=df)
plt.show()
```

### Step 5: Model Development

#### Question 5: Correlation Between `sqft_above` and `price`
Use `regplot` from Seaborn to visualize the correlation:

```python
sns.regplot(x='sqft_above', y='price', data=df)
plt.show()
```

#### Fit a Linear Regression Model with One Feature
Fit a model using `long` and calculate the R²:

```python
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
r_squared = lm.score(X, Y)
print("R² for 'long':", r_squared)
```

#### Question 6: Fit a Model with `sqft_living`
Fit a model with `sqft_living` as the feature:

```python
X = df[['sqft_living']]
lm.fit(X, Y)
r_squared = lm.score(X, Y)
print("R² for 'sqft_living':", r_squared)
```

#### Question 7: Fit a Model with Multiple Features
Use multiple features to fit a model:

```python
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features]
lm.fit(X, Y)
r_squared = lm.score(X, Y)
print("R² with multiple features:", r_squared)
```

### Step 6: Model Evaluation and Refinement

#### Question 8: Create a Pipeline for Polynomial Regression
Create a pipeline that includes scaling, polynomial transformation, and linear regression:

```python
pipeline = Pipeline([('scale', StandardScaler()), 
                     ('polynomial', PolynomialFeatures(include_bias=False)), 
                     ('model', LinearRegression())])

pipeline.fit(X, Y)
r_squared = pipeline.score(X, Y)
print("R² with pipeline:", r_squared)
```

#### Question 9: Ridge Regression
Split the data into training and testing sets, and then fit a Ridge regression model:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, Y_train)
r_squared = ridge_model.score(X_test, Y_test)
print("R² with Ridge regression:", r_squared)
```

#### Question 10: Polynomial Transform and Ridge Regression
Perform a second-order polynomial transform and apply Ridge regression:

```python
# Perform polynomial transform on training and testing data
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit Ridge regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train_poly, Y_train)
r_squared = ridge_model.score(X_test_poly, Y_test)
print("R² with Polynomial Ridge regression:", r_squared)
```

---

By following these steps, you should be able to work through the entire assignment. Let me know if you need any further explanation on any of the steps!
