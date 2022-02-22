# Movie-Recommendation Systems
## Summary
A recommender system is a system which predicts ratings a user might give to a specific item. The goal of a recommender system is to generate meaningful recommendations to a collection of users for items or products that might interest them. Recommender systems are one of the most successful and widespread application of machine learning technologies in business. You can find large scale recommender systems in retail, video on demand, or music streaming.

In this project, I have tried to built some recommendation systems that suggests movies based on movie contents, people's interests and combination of these two approach that combines user ratings and content of the movies. 
I have built Content-based recommendation system, User-based collaborative filtering, and Hybrid recommendations system. 

## Dataset
I used MoviLens ml-latest dataset. This dataset describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 27.753.444 ratings from 283.228 users and 1.108.997 tag applications across 58.098 movies. 

The data are contained in the files genome-scores.csv, genome-tags.csv, links.csv, movies.csv, ratings.csv and tags.csv.

You can dowload the dataset directly from https://grouplens.org/datasets/movielens/latest/ 

## Preprocessing and Exploratory Data Analysis
The following preprocessing steps were applied on the dataset:

* Cleaning movie titles and extracting released years from movie.csv
* Combining movies.csv, genome-scores.csv, and genome-tags.csv files and choosing first 100 tags for models after relevance score analysis
* Combining movies.csv and ratings.csv and choosing movies have more than 90 reviews and top 50% of users have rated movies since dataset are huge and to be able to avoid memory error and to be able to get improved results. After dropping both unpopular movies and inactive users, our reshaped ratings data has 24010731 ratings (was 27753444 at the beginning)

## Models
### Content-Based Recommendation System
Content-based Filtering is a Machine Learning technique that uses similarities in features to make decisions. For this we need to have a minimal understanding of the users’ preferences, so that we can then recommend new items with similar tags/keywords to those specified (or inferred) by the user.

In this project, I used cosine similarity scores to get similarities between tags feature. And then I applied stemming process on tags column. After that, I added popularity of movies and ratings by using IMDB's the weighted rating of each movie formula ![image.png](attachment:16ad175e-014d-4ea6-9eef-dec35f292f2e.png) to get improved movie recommendations.

### User-Based Collaborative Filtering
Colloborative filtering is a type of personalized recommendation strategy that identifies the similarities between users. 

Our Content based engine suffers from some severe limitations such as:

* It is only capable of suggesting movies which are close to a certain movie.
* It is not really personal in that it doesn't capture the personal tastes and biases of a user.

To overcome these limitations and to capture personal tastes, I used the Surprise library that used extremely powerful algorithms like Singular Value Decomposition (SVD) and baseline Algorithm to minimise RMSE (Root Mean Square Error). I compared them and SVD gave me better results. 

### Hybrid Recommendation System 
A hybrid recommendation system is a special type of recommendation system which can be considered as the combination of the content and collaborative filtering method.

In this project, I took an user id as an input and I found the the movie title that the user like it (rated 5) since I found some other similar movies regarding its content. As an output, similar movies sorted on the basis of expected ratings by that particular user.

## Demo on Streamlit

