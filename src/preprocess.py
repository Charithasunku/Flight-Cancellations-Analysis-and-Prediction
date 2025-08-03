import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path):
    """
    Loads the flight data from a CSV file.
    """
    df = pd.read_csv(file_path)
    df.drop('Flight ID', axis=1, inplace=True)
    return df

def get_preprocessor(X):
    """
    Creates a preprocessor for numerical and categorical features.
    """
    categorical_features = ['Airline', 'Origin_Airport', 'Destination_Airport', 'Airplane_Type', 'Day_of_Week', 'Month', 'Scheduled_Departure_Time']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor
