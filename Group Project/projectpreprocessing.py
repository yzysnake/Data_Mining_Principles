import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


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


def apply_pca(df, n_components=2):
    # Separating the target variable and features
    X = df.drop('Attrition_Yes', axis=1)
    y = df['Attrition_Yes']

    # Applying PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Creating a DataFrame for the PCA results
    pca_columns = [f'PCA_Component_{i}' for i in range(1, n_components + 1)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)

    # Adding the target variable back to the PCA DataFrame
    df_pca['Attrition_Yes'] = y.reset_index(drop=True)

    return df_pca


def plot_learning_curves(model, X, y, cv):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=-1,
                                                            train_sizes=np.linspace(.1, 1.0, 10),
                                                            scoring='recall')

    # Recall measures the proportion of actual positive cases (employees who will leave) that are correctly
    # identified. High recall is important if the cost of missing an employee who is about to leave is high. For
    # example, if not identifying employees at risk of leaving means losing valuable talent and incurring significant
    # rehiring and training costs, you might prioritize recall.

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plotting the learning curves
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Drawing bands for the standard deviation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Creating plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Recall Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Swap the FP and TN in the confusion matrix to put TP in the top-right
    cm = cm[:, ::-1]
    cm = cm[::-1, :]

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Yes', 'No'], yticklabels=['Yes', 'No'])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()