# Nicholas Casas, 2001901158
# Brandon Timok, 
# CS 422 - Final Project

#imports
import pandas as pd
import numpy as np

#read in csv file
data = pd.read_csv('pokemon.csv')

#knn stuff 
k = 3

#function for euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))  #euclidean

