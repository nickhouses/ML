# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# Load the data from a CSV file
data = pd.read_csv('./pokemon.csv')

# Keep a copy of the original DataFrame for later use
original_data = data.copy()

# The target feature for predicting Pokemon type
type_target = data['type1']

# Drop irrelevant columns that won't be useful for the model
data = data.drop(columns=['pokedex_number', 'name', 'classfication', 'type1', 'type2'])

# Set up the classification problem
# The target variable is 'type1'
y = type_target

# The features are all other columns
X = data

# Fill NaN values with mean for numeric columns and 'unknown' for categorical columns
numeric_cols = X.select_dtypes(include='number').columns
categorical_cols = X.select_dtypes(include='object').columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
X[categorical_cols] = X[categorical_cols].fillna('unknown')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to predict a Pokémon's type given its index
def predict_pokemon(model, X_test, y_test, original_data, index):
    # Get the features for the selected Pokémon and convert to DataFrame
    pokemon_features = pd.DataFrame([X_test.loc[index]])

    # Predict the type for the selected Pokémon
    predicted_type = model.predict(pokemon_features)[0]

    # Get the actual type for the selected Pokémon
    actual_type = y_test.loc[index]

    # Get the name of the selected Pokémon
    pokemon_name = original_data.loc[index, 'name']

    # Print the results
    print(f"Pokémon: {pokemon_name}")
    print(f"Predicted type: {predicted_type}")
    print(f"Actual type: {actual_type}")

# Function to train and evaluate a K-Nearest Neighbors model
def knn_model(X_train, y_train, X_test, y_test, original_data):
    print("\n" + "-" * 50)
    print("K-Nearest Neighbors Model")
    print("-" * 50 + "\n")

    # Data Scaling and One-Hot Encoding
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Hyperparameter tuning
    params = {
        'regressor__n_neighbors': range(1, 10),
        'regressor__weights': ['uniform', 'distance'],
        'regressor__p': [1, 2]
    }
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', KNeighborsClassifier())])
    grid_search = GridSearchCV(pipeline, params, cv=5)
    grid_search.fit(X_train, y_train)

    # kNN regressor 
    knn = grid_search.best_estimator_

    # Fit the model to the training data
    knn.fit(X_train, y_train)

    # Predict the egg hatch times for the test data
    y_pred = knn.predict(X_test)

    # Calculate Accuracy and print classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred))

    return knn

# Function to train and evaluate a Random Forest model
def random_forest_model(X_train, y_train, X_test, y_test, original_data):
    print("\n" + "-" * 50)
    print("Random Forest Model")
    print("-" * 50 + "\n")

    # Data Scaling and One-Hot Encoding
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Hyperparameter tuning
    params = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 5, 10],
        'regressor__min_samples_split': [2, 5, 10]
    }
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', RandomForestClassifier(random_state=42))])
    grid_search = GridSearchCV(pipeline, params, cv=5)
    grid_search.fit(X_train, y_train)

    # Random Forest regressor 
    rf = grid_search.best_estimator_

    # Fit the model to the training data
    rf.fit(X_train, y_train)

    # Predict the egg hatch times for the test data
    y_pred = rf.predict(X_test)

    # Calculate Accuracy and print classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred))

    return rf

# Function to train and evaluate a Linear Regression model
def logistic_regression_model(X_train, y_train, X_test, y_test, original_data):
    print("\n" + "-" * 50)
    print("Logistic Regression Model")
    print("-" * 50 + "\n")

    # Data Scaling and One-Hot Encoding
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Linear Regression model with specified parameters
    lr = LogisticRegression()

    # Create a pipeline with preprocessor and LR model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lr)])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Predict the target variable for the test data
    y_pred = pipeline.predict(X_test)

    # Calculate Accuracy and print classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred))

    return pipeline
    
# Generate a random index for comparison
random_index = np.random.choice(X_test.index)

# Call the knn_model function to train and evaluate a K-Nearest Neighbors model
knn = knn_model(X_train, y_train, X_test, y_test, original_data)
predict_pokemon(knn, X_test, y_test, original_data, random_index)

# Call the random_forest_model function to train and evaluate a Random Forest model
rf = random_forest_model(X_train, y_train, X_test, y_test, original_data)
predict_pokemon(rf, X_test, y_test, original_data, random_index)

# Call the logistic_regression_model function to train and evaluate a Logistic Regression model
lr = logistic_regression_model(X_train, y_train, X_test, y_test, original_data)
predict_pokemon(lr, X_test, y_test, original_data, random_index)