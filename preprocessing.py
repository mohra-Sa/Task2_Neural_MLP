from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def preprocess_data():
    df = pd.read_csv("D:/3rd year/2nd term/Neural Network/Task2_neural/penguins.csv")

    # Handle missing values by species-wise mean
    numeric_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
    for col in numeric_cols:
        df[col] = df[col].fillna(df.groupby('Species')[col].transform('mean'))

    # Encode location as numeric
    location_mapping = {'Torgersen': 0.0, 'Biscoe': 1.0, 'Dream': 2.0}
    df['OriginLocation'] = df['OriginLocation'].map(location_mapping)

    # Features & target
    features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
    X = df[features].values
    y = df['Species'].values.reshape(-1, 1)

    # One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False)
    y = ohe.fit_transform(y)

    # Get species list in the same order as the one-hot columns
    species_list = ohe.categories_[0].tolist()

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,        # ← 40% = 20 test samples per class
        random_state=42,
        stratify=df['Species'] # ← preserves species ratio in both splits
    )

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  
    X_test  = scaler.transform(X_test)       

    # i need to print the dataset after preprocessing to verify it is correct
    print("Preprocessed X_train:\n", X_train)
    print("Preprocessed y_train:\n", y_train)
    print("Preprocessed X_test:\n", X_test)
    print("Preprocessed y_test:\n", y_test)
    
    # i need to print  the shapes of the datasets to verify they are correct
    print("Shapes after preprocessing:")
    print("X_train shape:", X_train.shape)  # Should be (90, 5)
    print("y_train shape:", y_train.shape)  # Should be (90, 3)
    print("X_test shape:", X_test.shape)    # Should be (60, 5)
    print("y_test shape:", y_test.shape)    # Should be (60, 3)

   



   



    #
    return X_train, y_train, X_test, y_test, scaler, species_list
