# Nicholas Casas, 2001901158
# Brandon Timok, 8000477724
# CS 422 - Final Project

# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# function for euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))  #euclidean

# Goals: 
# 1. Egg Hatch Times
# 2. Best Starting 6 

# read in csv file
data = pd.read_csv('Project/pokemon.csv')

# Step 1: Data Cleaning and Preprocessing

# Target feature for predicting egg hatch time
egg_target = data['base_egg_steps']

# Step 2: Feature Engineering

# Step 3: Set up the regression problem

# Target variable: base_egg_steps
y = data['base_egg_steps']

# Features will be all other columns except for 'base_egg_steps'
X = data.drop(columns=['base_egg_steps'])

# Drop non-numeric columns
X = X.select_dtypes(include='number')

# Calculate correlations between features and target
correlations = X.corrwith(y)

# Select the top 10 features with the highest correlation
top_features = correlations.abs().nlargest(10).index

# Use only the selected features
X = X[top_features]

# Drop irrelevant columns
X = X.drop(columns= ['against_bug', 'against_dark', 'against_dragon', 'against_electric', 
                           'against_fairy', 'against_fight', 'against_fire', 'against_flying', 
                           'against_ghost', 'against_grass', 'against_ground', 'against_ice', 
                           'against_normal', 'against_poison', 'against_psychic', 'against_rock', 
                           'against_steel', 'against_water','generation', 'is_legendary'])

# Fill NaN values with mean
X = X.fillna(X.mean())

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Implement KNN for regression

# Hyperparameter tuning
params = {'n_neighbors': range(1, 10)}
grid_search = GridSearchCV(KNeighborsRegressor(), params, cv=5)
grid_search.fit(X_train_scaled, y_train)

# kNN regressor 
knn = grid_search.best_estimator_

# Fit the model to the training data
knn.fit(X_train, y_train)

# Step 6: Make predictions 

# Predict the egg hatch times for the test data
y_pred = knn.predict(X_test)

# Prompt user for a Pokémon name
pokemon_name = input("Enter the name of a Pokémon: ")

# Find the Pokémon in the dataset
pokemon_data = data[data['name'].str.lower() == pokemon_name.lower()]

if len(pokemon_data) == 0:
    print(f"Error: Pokémon '{pokemon_name}' not found in the dataset.")
else:
    # Extract numerical features for the found Pokémon
    pokemon_features = pokemon_data[X.columns]  # Use same columns as X_train
    # pokemon_features = pokemon_features.fillna(X_train.mean())  # Fill missing values with X_train mean

    # Make sure to scale the Pokemon features before making predictions
    pokemon_features_scaled = scaler.transform(pokemon_features)

    # Make prediction for base egg steps using the kNN model
    predicted_base_egg_steps = knn.predict(pokemon_features)

    # Get the actual base_egg_steps from the dataset
    actual_base_egg_steps = pokemon_data['base_egg_steps'].values[0]

    # Print predicted and actual base egg steps
    print(f"Predicted base egg steps for {pokemon_name}: {predicted_base_egg_steps[0]:.0f}")
    print(f"Actual base egg steps for {pokemon_name}: {actual_base_egg_steps}")