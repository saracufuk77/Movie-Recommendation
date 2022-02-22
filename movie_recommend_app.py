import streamlit as st
import pickle
import pandas as pd
import numpy as np
import surprise

################################################################

################################################################
movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies_list = pd.DataFrame(movies_dict)
cosine_sim = pickle.load(open('contend_cosine_sim.pkl','rb'))

movies_list = movies_list.reset_index()
titles = movies_list['Title']
indices = pd.Series(movies_list.index, index=movies_list['Title'])

vote_counts = movies_list[movies_list['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies_list[movies_list['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.95)

def weighted_rating(x):
     v = x['vote_count']
     R = x['vote_average']

     return (v / (v + m) * R) + (m / (m + v) * C)

def recommend(title, n=5):
     idx = indices[title]
     sim_scores = list(enumerate(cosine_sim[idx]))
     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
     sim_scores = sim_scores[1:26]
     movie_indices = [i[0] for i in sim_scores]

     movies = movies_list.iloc[movie_indices][['Title', 'vote_count', 'vote_average']]
     vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
     vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
     C = vote_averages.mean()
     m = vote_counts.quantile(0.60)
     qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) &
                        (movies['vote_average'].notnull())]
     qualified['vote_count'] = qualified['vote_count'].astype('int')
     qualified['vote_average'] = qualified['vote_average'].astype('int')
     qualified['wr'] = qualified.apply(weighted_rating, axis=1)
     qualified = qualified.sort_values('wr', ascending=False).head(n)
     liste = qualified.values.tolist()

     recommended_movies = []
     for i in range(len(liste)):
          recommended_movies.append(liste[i][0])
     return recommended_movies
#############################################################################################

#############################################################################################
rating_dict = pickle.load(open('rating_dict.pkl','rb'))
user_list = rating_dict.values()
ratings = pickle.load(open('movies_data.pkl','rb'))
svd = pickle.load(open('colloborative_SVD_model.pkl', 'rb'))

def user_based_recommendation(user_id, n=5):
     iids = ratings['movieId'].unique()
     iids_user = ratings.loc[ratings['userId'] == user_id, 'movieId']
     iids_to_pred = np.setdiff1d(iids, iids_user)

     testset = [[user_id, iid, 4.0] for iid in iids_to_pred]
     predictions = svd.test(testset)

     pred_ratings = np.array([pred.est for pred in predictions])
     i_top_n = pred_ratings.argsort()[-n:][::-1]
     iids_topn = iids_to_pred[i_top_n]

     rec_films = pd.DataFrame()
     for i in iids_topn:
          rec_films = rec_films.append(ratings[ratings['movieId'] == i][:1])
     liste = rec_films.values.tolist()

     recommended_movies = []
     for i in range(len(liste)):
          recommended_movies.append(liste[i][1])
     return recommended_movies

#############################################################################################

############################################################################################
movie_rating = pickle.load(open('movie_list.pkl','rb'))
id_map = pickle.load(open('id_map.pkl','rb'))
indices_map = id_map.set_index('movieId')

def hybrid(userId,n=5):
    title = ratings.loc[ratings['userId']==userId,['Title', 'rating']].sort_values('rating', ascending=False)[:1]['Title'].values[0]
    idx = indices[title]
    tmdb_Id = id_map.loc[title]['tmdbId']
    movie_id = id_map.loc[title]['movieId']
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = movie_rating.iloc[movie_indices][['Title', 'vote_count', 'vote_average', 'movieId']]
    movies['est'] = movies['movieId'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['tmdbId']).est)
    movies = movies.sort_values('est', ascending=False).head(n)
    liste = movies.values.tolist()

    recommended_movies = []
    for i in range(len(liste)):
         recommended_movies.append(liste[i][0])
    return recommended_movies



###################################

###################################

st.title('Movie Recommender System')

pages_names= ['Content Based Recommendations','Colloborative Filtering Recommendations', 'Hybrid Recommendations']

page = st.radio('Choose Recommendation Type', pages_names)

if page == 'Content Based Recommendations':

     st.subheader('Content Based Recommendations')

     selected_movie_name = st.selectbox(' ',movies_list['Title'].values)

     if st.button('Recommend'):
          recommendations = recommend(selected_movie_name)
          for i in recommendations:
               st.write(i)

elif page == 'Colloborative Filtering Recommendations':

     st.subheader('Colloborative Filtering Recommendations')

     selected_user = st.selectbox('User Id ', user_list)

     if st.button('Recommend'):
          recommendations = user_based_recommendation(selected_user)
          for i in recommendations:
               st.write(i)

else:

     st.subheader('Hybrid Recommendations')

     selected_user = st.selectbox('User Id ', user_list)

     if st.button('Recommend'):
          recommendations = hybrid(selected_user)
          for i in recommendations:
               st.write(i)

