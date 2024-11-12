import pandas as pd

#reading the CSV files, change them on your personal computer to wherever you saved them
movies = pd.read_csv(r'C:\Users\josea\Documents\College\CompSci\Machine Learning\Movie Recs\ml-latest-small\movies.csv')
ratings = pd.read_csv(r'C:\Users\josea\Documents\College\CompSci\Machine Learning\Movie Recs\ml-latest-small\ratings.csv')
tags = pd.read_csv(r'C:\Users\josea\Documents\College\CompSci\Machine Learning\Movie Recs\ml-latest-small\tags.csv')
userID = ratings
#dropping timestamp as it's unnecessary
tags = tags.drop(columns='timestamp')
ratings = ratings.drop(columns='timestamp') 

#Groups on userId and movieId and groups tags into a list
tags = tags.groupby(['movieId']).agg({'tag':set}).reset_index()

#Refining ratings to just be movieId and the average rating to only 1 decimal point
ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
ratings = ratings.rename(columns={'rating':'average_rating'})
ratings['average_rating'] = ratings['average_rating'].round(1)
filtered_data = pd.merge(movies,ratings, on='movieId')
filtered_data = pd.merge(filtered_data, tags, on='movieId')

filtered_data.to_csv('filtered_movies.csv',index=False)

#For seeing changes
print(movies.head())
print(ratings.head())
print(tags.head())
print(filtered_data)