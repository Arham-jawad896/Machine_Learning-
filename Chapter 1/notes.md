# Chapter 2

In this chapter, we create a very simple project where we predict house prices given the features such as Total Number of Bedrooms, age of the house, size of the house, and much much more.

## Pipeline

A pipeline is just a set of data processing components. Pipelines are used very commonly as there are lot of data transformations to apply.

## Classification or Regression?

First, we need to decide whether this problem is Regression or Classification. In Classification, your model chooses from a given set of categories, like Cat or Dog? Genre of a Movie. In classification, the options are limited, the model just has to predict the probability of a specific thing being in a class, so it has to choose. In Regression, your model has to predict a continuous value, meaning it doesn't choose, the number it predicts can go upto infinity and down to negative infinity. The Housing Price problem is a Regression problem as we are predicting a price, and prices can vary highly.

## Select a Performance Measure

Now that we know what kind of problem it is, we need to select a performance measure. For regression, the most commonly used performance measure is the RMSE, which tells you how bad your model is, not how good it is, so the goal in this problem is to minimize your RMSE, not maximize it like accuracy.

## Take a Look at the Data Structure

You can start by looking at the top five rows of the data using panda's head() method.

Note: You can look at more than 5 rows by specifying a number inside the head() method's bracket, the default value is 5 rows.

## Create a Test Set

In Machine Learning, your models learn from a training set, and then you test them on the test set, so at this point, we will create a test set, a typical split is 80/20, where we set 80% of data for training and remaining data for testing.

Scikit-learn has a `train_test_split` method inside its `sklearn.model_selection` library. This allows you to specify your features, and your labels, and your test_size, and it automatically splits it for you:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Data Cleaning

Machine Learning Algorithms cannot work with missing features, so you have to take care of them, for this, you can either drop the entire rows containing missing values, drop the full feature, or set the missing values to some other value like mean, median, mode. The third option is the best and most commonly used one. Scikit-learn provides a `SimpleImputer` inside its `sklearn.impute` library:

```python
from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy="median")
```

Since you can only apply median on numerical features, lets set aside a subset of our dataset containing only the numerical attributes:

```python
import numpy as np

housing_num = housing.select_dtypes(include=[np.number])

X = imputer.fit_transform(housing_num)
```

## Handling Text and Categorical Attributes

Since ML Models cannot work with categorical data, we have to convert them into numbers. To do this, we can use Ordinal Encoder, which assigns a number to each category, it is useful for tasks where two nearby categories are more similar than two distant categories, like "Bad", "Average", "Good", "Excellent" etc. But in our scenerio, OneHotEncoder is better. It creates one column for each category, so in the presence of one category, the column for that category will be 1 and other will be 0 and vice versa:

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```

## Feature Scaling

ML Models cannot work with numbers on different scales like 1, and 100,000 or 100,000 and 1,000,000,000, so we need to balance them and bring them on a similar scale like between 0 and 1. To do this, Scikit-learn provides a `StandardScaler` class in the `sklearn.preprocessing` library:

```python
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```

## Transformation Pipelines

As you can clearly see, you need to apply a lot of transformations on your data, like Imputing, Feature Scaling, Categorical Encoding, and doing them manually one by one seems redundant. So, we can use the `Pipeline` class provided by `sklearn.pipeline` library to group all these data transformations together:

```python
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
```
It would be more convenient to handle both categorical and numerical features together, for this, we can use the `ColumnTransformer` class provided by the `sklearn.compose` library:

```python
from sklearn.compose import ColumnTransformers

num_attribs = housing.select_dtypes(include=[np.number]).columns
cat_attribs = housing.select_dtpes(include=[np.object]).columns

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehotencode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
```

## Select and Train a Model

Finally! Its time for us to select and train a machine learning model on our data! In our case, we have used the XGBoost Regressor which has same syntax as a RandomForestRegressor:

```python
XGBRegressor(
        n_estimators=330,
        max_depth=10,
        random_state=42
    )
```
## Evaluate on the Test Set

I evaluated our model on the test set, and here are the results, not bad!

```
RMSE: 764340224.0, RMSE: 27646.703125
```
