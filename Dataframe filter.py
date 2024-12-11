import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np

#Block One: First Filtering
#reading the CSV files, change them on your personal computer to wherever you saved them
rate = pd.read_csv(r'C:\Users\josea\Documents\College\CompSci\Machine Learning\Movie Recs\ml-latest-small\ratings.csv')
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

#Block Two: Unique Genres and Second Filtering
# Split genres by '|' and get all unique genres
unique_genres = set()
for genres in movies['genres']:
    unique_genres.update(genres.split('|'))
unique_genres = sorted(unique_genres)  # Sort in alphabetic order
for genre in unique_genres:
    filtered_data[genre] = filtered_data['genres'].apply(lambda x: 1 if genre in x.split('|') else 0)
#filter out In Netflix Queue
filtered_data['tag'] = filtered_data['tag'].apply(lambda tag_set: {tag for tag in tag_set if tag != "In Netflix queue"})

# Drop the original 'genres' column (I think it's a good idea, no need to clutter the data yk?)
filtered_data= filtered_data.drop(columns=['genres'])


#Block Three: Vectorization
#I changed the vector size to 10 just cause it's smaller but if you wanna increase it by all means go for it
wordVector = Word2Vec(filtered_data['tag'].tolist(),vector_size=10, window=5, min_count=1, workers=4)

vector_size = 10  # Same size as Word2Vec vectors
padding_vector = np.zeros(vector_size)
# Function to generate a vector for up to 3 tags
def create_feature_vector(row):
    # Get word vectors for the tags, up to 3 tags, and pad if fewer
    tag_vectors = [
        wordVector.wv[tag] if tag in wordVector.wv else padding_vector
        for tag in list(row['tag'])[:3]
    ]
    while len(tag_vectors) < 3:  # Pad with zero vectors if fewer than 3 tags
        tag_vectors.append(padding_vector)
    
    # Flatten the tag vectors (3 vectors of size 10 each -> 30 elements)
    tag_vector = np.concatenate(tag_vectors)
    
    # Add genre one-hot encoding
    genre_vector = row[unique_genres].values  # One-hot encoded genres
    
    # Combine tag vector and genre vector
    feature_vector = np.concatenate([tag_vector, genre_vector])
    return feature_vector
filtered_data['feature_vector'] = filtered_data.apply(create_feature_vector, axis=1)

#Okay so I mixed the tag vector and the genre vectors into one feature vector
#I am going to drop the original tag vector and the genres
filtered_data = filtered_data.drop(columns=list(unique_genres))
filtered_data = filtered_data.drop(columns='tag')

#wrote to file to see feature data
#filtered_data.to_csv('filtered_movies.csv',index=False)

#Block 4: The actual recommendation

#TODO: implement the test and training sets split

#Query is the movie we want to isolate
#Cosine Similarities
def knn_recommendation_cos(query_vector,train_data,watched_movies,k=5):

    # Extract feature vectors and movieIds
    movie_ids = train_data['movieId'].values
    feature_vectors = np.vstack(train_data['feature_vector'].values)

    # Calculate cosine similarities
    similarities = cosine_similarity([query_vector], feature_vectors).flatten()

    # Combine movieIds with similarities
    recommendations = list(zip(movie_ids, similarities))

    # Sort by similarity score in descending order
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    # Exclude movies the user has already watched
    recommendations = [rec for rec in recommendations if rec[0] not in watched_movies]

    # Remove duplicates by keeping only the first occurrence of each movieId
    seen_movies = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec[0] not in seen_movies:
            unique_recommendations.append(rec)
            seen_movies.add(rec[0])

    # Take top-k unique recommendations
    top_k = unique_recommendations[:k]

    # Create a DataFrame for the top-k recommendations
    recommended_movie_ids = [rec[0] for rec in top_k]
    recommended_scores = [rec[1] for rec in top_k]

    # Removes already recommended movies by movieId 
    recommended_movies = train_data[train_data['movieId'].isin(recommended_movie_ids)]
    recommended_movies = recommended_movies.drop_duplicates(subset=['movieId'])
    recommended_movies = recommended_movies.set_index('movieId').loc[recommended_movie_ids].reset_index()
    recommended_movies['similarity_score'] = recommended_scores

    return recommended_movies

#if you want to compare cosine to euclidean distance
def knn_recommendation_eucl(query,train_data,k=5):
    #Extract ratings and feature vectors from training set
    #Ratings are optional if you don't want them, I have them here to use as bias
    train_features = np.array(train_data['feature_vector'].tolist())
    train_ratings = train_data['average_rating'].values#Optional

    #kNN using euclidean distances
    distances = euclidean_distances([query],train_features).flatten()

    #Optional Bias
    weighted_distances= distances / train_ratings

    #get the indicies of nearest movies (5)
    kNN_indices = np.argsort(weighted_distances)[:k] #Sorted by weighted distances ascending order

    #Get the movies from the index
    kNN_movies = train_data.iloc[kNN_indices]
    return kNN_movies


def my_train_test_split(df, filt_data, n=7):
    # Create empty lists to store train and test data
    train_list = []
    test_list = []

    # Group by 'userID' to process each user individually
    df.drop(columns='timestamp')
    merged_df = pd.merge(df, filt_data, on='movieId')
    grouped = merged_df.groupby('userId')

    for _, group in grouped:
        # Sort movies by rating in descending order
        sorted_group = group.sort_values(by='rating', ascending=False)
        
        # Select the top n rated movies for testing
        test = sorted_group.head(n)
        
        # Use the rest for training
        train = sorted_group.iloc[n:]
        
        # Append to respective lists
        test_list.append(test)
        train_list.append(train)

    # Combine all train and test splits into DataFrames
    train_data = pd.concat(train_list).reset_index(drop=True)
    test_data = pd.concat(test_list).reset_index(drop=True)

    # print("Training Data:")
    # print(train_data)
    # print("\nTesting Data:")
    # print(test_data)
    # print(type(train_data))

    return train_data, test_data

# train_data, test_data = my_train_test_split(rate, filtered_data,n=0)
train_data1, test_data1 = my_train_test_split(rate, filtered_data,n=5)

def finalResults():
    #this is a hardcoded test set, if you want to implement your test data try using a for loop to iterate through it
    query_vector = test_data1.loc[test_data1['movieId'] == 4878, 'feature_vector'].values[0]
    print(test_data1.loc[test_data1['movieId']==4878,'title'].values[0])
    # print(query_vector)
    #calling the recommendation service based on cosine similarity with similarity scores
    recommended_movies = knn_recommendation_cos(query_vector, train_data1, rate[rate['userId'] == 610]['movieId'].values)
    # print(recommended_movies)
    #clean up the output by dropping user id's ratings of specific user and timestamp
    recommended_movies = recommended_movies.drop(columns=['userId','rating','timestamp'])

    return recommended_movies
print(finalResults())

