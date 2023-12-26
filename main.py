import sys
import lightgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint
from sklearn.model_selection import (
    cross_validate,
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import joblib

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 5000)

df_train = pd.read_csv("data.csv")
df_train.head()

df_train.columns
df_train.info()


value_to_delete = 0
num_rows_to_delete = 200000

indices_to_delete = df_train[df_train["HeartDiseaseorAttack"] == value_to_delete].index
rows_to_delete = np.random.choice(
    indices_to_delete,
    size=min(num_rows_to_delete, len(indices_to_delete)),
    replace=False,
)

df_deleted = df_train.loc[rows_to_delete]
df_deleted.to_csv("heart0.csv")

df = df_train.drop(rows_to_delete)

df["HeartDiseaseorAttack"].value_counts()


####### Visualize

sns.set(style="whitegrid")

plt.figure(figsize=(18, 12))

for i, column in enumerate(df.columns):
    plt.subplot(5, 5, i + 1)
    sns.histplot(df[column], kde=True)
    plt.title(column)

plt.tight_layout()
plt.savefig("distribution.png", bbox_inches="tight")
plt.show()


# Correlation
corr_matrix = df.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.savefig("correlation.png", bbox_inches="tight")
plt.show()


# Boxplot

plt.figure(figsize=(18, 12))

for i, column in enumerate(df.columns):
    plt.subplot(5, 5, i + 1)
    sns.boxplot(x="Sex", y=column, data=df)
    plt.title(column)

plt.tight_layout()
plt.savefig("boxplot.png", bbox_inches="tight")
plt.show()


df.describe().T

# Categorical

plt.figure(figsize=(18, 12))

for i, column in enumerate(df.columns):
    if len(df[column].unique()) < 10:  # Kategorik değişkenler için
        plt.subplot(5, 5, i + 1)
        sns.countplot(x=column, data=df)
        plt.title(column)

plt.tight_layout()
plt.savefig("categorical.png", bbox_inches="tight")
plt.show()


# Bar grafiği oluştur
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x=df["HeartDiseaseorAttack"].map({0:"Clean", 1:"Sick"}), hue = df["Sex"].map({0:"Female", 1:"Male"}))
plt.xlabel('Cinsiyet')
plt.ylabel('Hasta Sayısı')
plt.legend()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x=df["Income"], hue = df["Sex"].map({0:"Kadın", 1:"Erkek"}))
plt.xlabel('Gelir Durumu (Her bir değer belirli bir skalayı belirtmektedir.)')
plt.ylabel('Kişi Sayısı')
plt.legend()







def check_df(dataframe, head=5, tail=5):
    """
    Display various information about the DataFrame, including shape, data types, head, tail, missing values, and quantiles.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        head : int, optional
            Number of rows to display for the head. Default is 5.
        tail : int, optional
            Number of rows to display for the tail. Default is 5.

    Returns
    ------
        None

    Prints the following information:
    - Shape of the DataFrame.
    - Info about the DataFrame, including data types, non-null counts, and memory usage.
    - First 'head' rows of the DataFrame.
    - Last 'head' rows of the DataFrame.
    - Number of missing values for each column.
    - Quantiles (0%, 5%, 50%, 95%, 99%, 100%) for numerical columns with at least 10 unique values.

    Examples
    ------
        # Display information for the entire DataFrame (default head=5, tail=5)
        check_df(df)

        # Display information with a custom number of rows for head and tail
        check_df(df, head=10, tail=15)
    """

    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.info())
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    numerical_cols = [
        col
        for col in dataframe.columns
        if dataframe[col].dtype in ["int64", "float64"]
        and dataframe[col].nunique() >= 10
    ]
    print(dataframe[numerical_cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identify and categorize columns in a DataFrame based on their data types and cardinality.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        cat_th : int, optional
            Numerical threshold for considering a numerical variable as categorical. Default is 10.
        car_th : int, optional
            Categorical threshold for considering a categorical variable as cardinal. Default is 20.

    Returns
    ------
        cat_cols : list
            List of categorical variables.
        num_cols : list
            List of numerical variables.
        cat_but_car : list
            List of categorical variables with cardinality exceeding 'car_th'.
        num_but_cat : list
            List of numerical variables with cardinality below 'cat_th'.

    Examples
    ------
        # Example: Categorize and display column information for a DataFrame
        cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=8, car_th=15)
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    """
    Calculate the lower and upper bounds for identifying outliers in a numerical column.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        col_name : str
            The name of the numerical column for which outlier thresholds will be calculated.
        q1 : float, optional
            The lower quartile value. Default is 0.10.
        q3 : float, optional
            The upper quartile value. Default is 0.90.

    Returns
    ------
        low_limit : float
            The lower threshold for identifying outliers.
        up_limit : float
            The upper threshold for identifying outliers.

    Examples
    ------
        # Example: Calculate outlier thresholds for a numerical column
        low_limit, up_limit = outlier_thresholds(df, 'numeric_column')
    """

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.10, q3=0.90):
    """
    Check for outliers in a numerical column of a DataFrame based on custom quantiles.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        col_name : str
            The name of the numerical column to check for outliers.
        q1 : float, optional
            The lower quantile value for calculating the lower threshold. Default is 0.10.
        q3 : float, optional
            The upper quantile value for calculating the upper threshold. Default is 0.90.

    Returns
    ------
        str
            A string indicating the presence of outliers in the specified numerical column.

    Examples
    ------
        # Example: Check for outliers in a numerical column with custom quantiles
        result = check_outlier(df, 'numeric_column', q1=0.05, q3=0.95)
        print(result)

        # Example 2: Check for outliers for multiple numerical columns
        for col in num_cols:
            print(check_outlier(df, col))
    """

    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    print(f"{col_name} Lower Limit: {low_limit}, {col_name} Upper Limit: {up_limit}")

    outliers = dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ]

    num_outliers = outliers.shape[0]  # Count the number of outliers

    if num_outliers > 0:
        return f"{col_name} : {num_outliers} : True"
    else:
        return f"{col_name} : {num_outliers} : False"




for col in num_cols:
    check_outlier(df, col)


def replace_with_thresholds(dataframe, variable, q1=0.10, q3=0.90):
    """
    Replace values in a numerical column with their respective outlier thresholds based on custom quantiles.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        variable : str
            The name of the numerical column for which outliers will be replaced with their thresholds.
        q1 : float, optional
            The lower quantile value for calculating the lower threshold. Default is 0.10.
        q3 : float, optional
            The upper quantile value for calculating the upper threshold. Default is 0.90.

    Returns
    ------
        None

    Modifies the DataFrame in-place by replacing values below the lower threshold with the lower threshold,
    and values above the upper threshold with the upper threshold.

    Examples
    ------
        # Example: Replace outliers in a numerical column with custom quantiles
        replace_with_thresholds(df, 'numeric_column', q1=0.05, q3=0.95)

        # Example 2: Replace outliers for multiple numerical columns
        for col in num_cols:
            replace_with_thresholds(df, col)
    """

    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)

    # Explicitly cast the limits to the data type of the column
    low_limit = low_limit.astype(dataframe[variable].dtype)
    up_limit = up_limit.astype(dataframe[variable].dtype)

    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



replace_with_thresholds(df, "BMI")


def target_summary_with_num(dataframe, target, num_col):
    """
    Generate a summary for a numerical column grouped by a target variable in a DataFrame.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        target : str
            The name of the target variable by which the numerical column will be grouped.
        num_col : str
            The name of the numerical column for which the summary is generated.

    Returns
    ------
        None

    Prints the mean of the specified numerical column grouped by the target variable.

    Examples
    ------
        # Example 1: Summary for a numerical column grouped by a binary target variable
        target_summary_with_num(df, 'target_variable', 'numeric_column')

        # Example 2: Summary for multiple numerical columns
        for col in num_cols:
            target_summary_with_num(df, 'target_variable', col)
    """

    print(dataframe.groupby(target).agg({num_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, cat_col):
    """
    Generate a summary for a categorical column in relation to a target variable in a DataFrame.

    Parameters
    ------
        dataframe : pd.DataFrame
            The input DataFrame.
        target : str
            The name of the target variable.
        cat_col : str
            The name of the categorical column for which the summary is generated.

    Returns
    ------
        None

    Prints a DataFrame with the mean of the target variable for each category in the specified categorical column.

    Examples
    ------
        # Example 1: Summary for a categorical column in relation to a binary target variable
        target_summary_with_cat(df, 'target_variable', 'categorical_column')

        # Example 2: Summary for multiple categorical columns
        for col in cat_cols:
            target_summary_with_cat(df, 'target_variable', col)
    """

    print(
        pd.DataFrame(
            {
                f"{target}_MEAN": dataframe.groupby(cat_col, observed=False)[
                    target
                ].mean()
            }
        ),
        end="\n\n\n",
    )
    print("#######################################", end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "HeartDiseaseorAttack", col)

df["Income"].value_counts()

df["Income"]



#Feature Engineering

df.head()

#araliklar = [18, 25, 30, 35, df["BMI"].max()]
#df["BMI_cat"] = pd.cut(df["BMI"], bins = araliklar, labels = ["Underweight", "Normalweight", "Overweight", "Obese"])


df["BMI"].describe()


df.groupby("Income").agg({"NoDocbcCost": "mean"})

#df.loc[(df["Income"] <= 4), "Income_Cat"] = "Poor"
#df.loc[(df["Income"] >= 5) & (df["Income"] < 8), "Income_Cat"] = "Normal"
#df.loc[(df["Income"] >= 8), "Income_Cat"] = "Rich"

#df['HealthyLifestyleScore'] = (df['PhysActivity'] + df['Fruits'] + df['Veggies'] - df['Smoker'] - df['HvyAlcoholConsump'])

risk_factors = ['HighBP', 'HighChol', 'Smoker', 'Diabetes', 'HvyAlcoholConsump']
df['RiskFactorCount'] = df[risk_factors].sum(axis=1)
df['OverallHealthScore'] = df['GenHlth'] + df['MentHlth']


"""

bins = [0, 2, 4, 6, 8, 10, 12]
labels = ['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High', 'Very High']
df['SES'] = pd.cut(df['Education'] + df['Income'], bins=bins, labels=labels, right=False)
#Sosyoekonomik statü


df['UnhealthyLifestyle'] = df[['Smoker', 'HvyAlcoholConsump', 'NoDocbcCost']].any(axis=1).astype(int)

df['DietaryHabitsScore'] = df['Fruits'] + df['Veggies']

"""

df.head()

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)



#Encoding
df_model_train = df.copy()
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype="int")
    return dataframe

df_model_train = one_hot_encoder(df, cat_cols, True)


#MODEL
X = df_model_train.drop("HeartDiseaseorAttack_1.0", axis=1)
y = df_model_train["HeartDiseaseorAttack_1.0"]


scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])



def base_models(x, y, scoring="roc_auc"):
  print("Base Models....")
  classifiers = [("LR", LogisticRegression(max_iter = 1000)),
                 ("KNN", KNeighborsClassifier()),
                 ("SVC", SVC()),
                 ("CART", DecisionTreeClassifier()),
                 ("RF", RandomForestClassifier()),
                 ("XGBoost", XGBClassifier(use_label_encoder = False, eval_metric = "logloss")),
                 ("LightGBM", LGBMClassifier(force_row_wise=True))]

  for name, classifier in classifiers:
    cv_results = cross_validate(classifier, x, y, cv=3, scoring = scoring)
    print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


base_models(X, y)





def preprocess_data(df, cat_cols, num_cols):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_processed = df.copy()

    # Feature engineering
    risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
    df_processed["RiskFactorCount"] = df_processed[risk_factors].sum(axis=1)
    df_processed["OverallHealthScore"] = (
        df_processed["GenHlth"] + df_processed["MentHlth"]
    )
    cat_cols, num_cols, cat_but_car, cat_but_num = grab_col_names(df_processed)

    cat_cols.remove("HeartDiseaseorAttack")
    df_encoded = pd.get_dummies(
        df_processed, columns=cat_cols, drop_first=True, dtype="int"
    )

    X = df_encoded.drop("HeartDiseaseorAttack", axis=1)
    y = df_encoded["HeartDiseaseorAttack"]

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y











