import pandas as pd
from sklearn.preprocessing import StandardScaler
pd.set_option("display.width", 5000)
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



data = {
  "HighBP": 0,
  "HighChol": 0,
  "CholCheck": 0,
  "BMI": 28,
  "Smoker": 0,
  "Stroke": 0,
  "Diabetes": 0,
  "PhysActivity": 0,
  "Fruits": 1,
  "Veggies": 1,
  "HvyAlcoholConsump": 0,
  "AnyHealthcare": 1,
  "NoDocbcCost": 0,
  "GenHlth": 1,
  "MentHlth": 0,
  "PhysHlth": 0,
  "DiffWalk": 0,
  "Sex": 1,
  "Age": 1,
  "Education": 5,
  "Income": 6
}

def preprocess_data(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_processed = pd.DataFrame([df])
    print(df_processed)
    # Feature engineering
    risk_factors = ["HighBP", "HighChol", "Smoker", "Diabetes", "HvyAlcoholConsump"]
    df_processed["RiskFactorCount"] = df_processed[risk_factors].sum(axis=1)
    df_processed["OverallHealthScore"] = (
        df_processed["GenHlth"] + df_processed["MentHlth"]
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(df_processed)

    return X










