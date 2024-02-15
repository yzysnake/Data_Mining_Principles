import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def one_hot_encode(df):
    # Specify the columns to be one-hot encoded
    columns_to_encode = ['Department', 'MaritalStatus', 'Gender', 'JobRole',
                         'EducationField', 'Attrition', 'Over18', 'OverTime', 'BusinessTravel']

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first', dtype=np.integer)

    # Fit and transform the specified columns
    encoded_data = encoder.fit_transform(df[columns_to_encode])

    # Get new column names for the one-hot encoded variables
    encoded_columns = encoder.get_feature_names_out(columns_to_encode)

    # Create a DataFrame with the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

    # Drop the original columns from the DataFrame
    df_dropped = df.drop(columns=columns_to_encode)

    # Concatenate the original DataFrame (minus the to-be-encoded columns) with the new one-hot encoded DataFrame
    df_encoded = pd.concat([df_dropped, encoded_df], axis=1)

    return df_encoded


def min_max_scale(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled
