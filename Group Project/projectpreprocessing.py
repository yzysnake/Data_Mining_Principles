from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd


def encode_nominal_default(df):
    """
    Encodes predefined nominal variables using one-hot encoding.

    Parameters:
    - df: pandas DataFrame.

    Returns:
    - DataFrame with one-hot encoded nominal variables.
    """
    nominal_columns = ['Department', 'MaritalStatus', 'Gender', 'JobRole', 'EducationField']  # Predefined nominal columns
    df_nominal = pd.get_dummies(df[nominal_columns], drop_first=True)
    return df_nominal


def encode_ordinal_default(df):
    """
    Encodes the predefined ordinal variable and binary nominal variables according to their mappings.

    Parameters:
    - df: pandas DataFrame.

    Returns:
    - DataFrame with the encoded ordinal and binary nominal variables.
    """
    # Predefined ordinal column and mapping
    ordinal_column = 'BusinessTravel'
    ordinal_mapping = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    df[ordinal_column] = df[ordinal_column].replace(ordinal_mapping)

    # Mapping for binary nominal variables
    binary_nominal_mapping = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}
    df['Over18'] = df['Over18'].replace(binary_nominal_mapping)
    df['OverTime'] = df['OverTime'].replace(binary_nominal_mapping)

    # Returning the DataFrame with the encoded columns
    return df[['BusinessTravel', 'Over18', 'OverTime']]


def combine_encoded_features_default(df, df_encoded_nominal, df_encoded_ordinal):
    """
    Combines the original DataFrame with the encoded DataFrames, excluding the original columns that were encoded.

    Parameters:
    - df: original pandas DataFrame.
    - df_encoded_nominal: DataFrame containing encoded nominal variables.
    - df_encoded_ordinal: DataFrame containing the encoded ordinal variable.

    Returns:
    - DataFrame with combined features.
    """
    exclude_columns = ['Department', 'MaritalStatus', 'Gender', 'JobRole', 'EducationField']  # Predefined columns to exclude
    df_combined = pd.concat([df.drop(columns=exclude_columns), df_encoded_nominal, df_encoded_ordinal], axis=1)
    return df_combined
