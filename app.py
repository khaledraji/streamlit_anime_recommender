# IMPORT
import streamlit as st

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
#import zipfile, os
#from zipfile import ZipFile
#import codecs
#from bs4 import BeautifulSoup
import pickle

vectorizer = TfidfVectorizer(stop_words='english')

# FUNCTIONS
#st.set_page_config(layout="wide")
@st.cache(allow_output_mutation=True)
def load_data(data):
	df = pd.read_csv(data)
	return df

def recommend(genres, data, n):
    tfidf =vectorizer.fit_transform(data).toarray()
    tfidf_test =vectorizer.transform([genres]).toarray()
    sim = cosine_similarity(tfidf_test, tfidf)
    anime_list = np.argsort(sim)[0][-100:] #sorted(list(enumerate(sim)), reverse=True, key=lambda x: x[1])[1:11]
    df = anime.loc[anime_list]
    df = df.sort_values(by=['Score'], ascending=False)

    return df.head(n)


def get_anime_name(aid, anime_df):
    try:
        name = anime_df.loc[anime_df.MAL_ID == aid]['English name'].values[0]
    except:
        name = "Unknown"
    return name

def get_recos(my_anime, n=10):
    recos = []
    id = (anime[anime['English name'] ==my_anime]['MAL_ID']).values[0]
    idx = aid_to_idx[id]
    sims_idx = similarity_scores[idx]
    for idx in np.argsort(-sims_idx):
        aid = idx_to_aid[idx]
        name = get_anime_name(aid, anime)
        genres = (anime[anime['English name'] ==name]['Genres']).values[0]
        episodes = (anime[anime['English name'] ==name]['Episodes']).values[0]
        syn = (anime[anime['English name'] ==name]['sypnopsis']).values[0]
        score = (anime[anime['English name'] ==name]['Score']).values[0]
        image = (anime[anime['English name'] ==name]['Url_image']).values[0]
        source = (anime[anime['English name'] ==name]['Url_source']).values[0]

        recos.append((aid,name, genres, episodes, syn, score, image, source))
    return recos[1:n+1]

def cut_text(text):
	if len(text) > 200:
		text = text[:200]+'[...]'
	else:
		text = text
	return text

genres_list = ['Action',
'Adventure',
'Shounen',
'Comedy',
'Drama',
'Fantasy',
'Samurai',
'Military',
'Historical',
'Horror',
'Mystery',
'Romance',
'Sci-Fi',
'Kids',
'Slice of Life',
'Sports',
'Supernatural',
'Super Power',
'Parody',
'Police']

# LOAD DATA AND FEATURE ENGINEERING

#Load the data we cleaned previously
anime = load_data("final_anime_df.csv")

#Load the results of our model trained with LighFM
similarity_scores = pickle.load(open("similarity_scores.pkl", "rb"))
idx_to_aid = pickle.load(open("idx_to_aid.pkl", "rb"))
aid_to_idx = pickle.load(open("aid_to_idx.pkl", "rb"))


# STREAMLIT APP
anime_names = anime['English name']

st.title("Anim'Mate Recommender App")
#st.title('Anime Recommendation App')
#st.image('https://static.hitek.fr/img/actualite/2015/07/fb_y2dlyefcp6rcmtixivkmejaowis.webp')
st.image('https://media.techtribune.net/uploads/2021/04/anime-day-1264669-1280x0-1021x580.jpeg')
st.subheader('Choose your model')
models = st.radio("If you already know some Animes, choose the first option. If you are neophyte, choose your favourite genres.",('Based on your favourite Anime', 'Based on Genres'))
if models == 'Based on your favourite Anime':
	st.subheader('Select an Anime')
	option = st.selectbox('',(anime_names))
	cola, colb, colc = st.columns(3)
	with colb:
		st.write('You selected:', option)
		id = (anime[anime['English name'] ==option]['MAL_ID']).values[0]
		url = (anime[anime['MAL_ID'] ==id]['Url_image']).values[0]
		st.image(url)
		slider = st.slider("How many recommendations ?", 1, 20, 10)
		recommender = st.button('Get Recommendation')
	if recommender:
		result = get_recos(option, n=slider)
		st.write('If you liked '+ option + ', then you might like the following Animes : ')
		for item in result:
			rec_title = item[1]
			rec_genre = item[2]
			rec_ep = item[3]
			rec_syn = item[4]
			rec_score = round(item[5],1)
			rec_image = item[6]
			rec_source = item[7]
			#col5, padding,col6, padding,col7 = st.columns((15,5,20,5,20))
			col5,col6,col7 = st.columns(3)
			with col5:
				st.image(rec_image, width=200)

			with col6:
				st.subheader(rec_title)
				st.write('Score : ',str(rec_score)+' /🔟')
				st.write('Genres 🏷️ : ',rec_genre)
				st.write('Episodes ⏳ : ',rec_ep)
				#url = rec_source
				#st.write("ℹ️ More info [here](%s)" % url)
				#st.write('Quick look ▶️ 📺 : ',)

			with col7:
				st.write('')
				st.write('Synopsis 📜 :')
				st.write(cut_text(rec_syn))
				st.write('')
				url = rec_source
				st.write("ℹ️ More info [here](%s)" % url)
else:
	st.write('Based on the genres you like we will give you Animes to binge watch')
	input = ''

	st.subheader('Select the genres you like')

	col1, col2, col3, col4 = st.columns(4)

	with col1:
		check = st.checkbox(genres_list[0], key = 0)
		check1 = st.checkbox(genres_list[1], key = 1)
		check2 = st.checkbox(genres_list[2], key = 2)
		check3 = st.checkbox(genres_list[3], key = 3)
		check4 = st.checkbox(genres_list[4], key = 4)

	with col2:
		check5 = st.checkbox(genres_list[5], key = 5)
		check6 = st.checkbox(genres_list[6], key = 6)
		check7 = st.checkbox(genres_list[7], key = 7)
		check8 = st.checkbox(genres_list[8], key = 8)
		check9 = st.checkbox(genres_list[9], key = 9)

	with col3:
		check10 = st.checkbox(genres_list[10], key = 10)
		check11 = st.checkbox(genres_list[11], key = 11)
		check12 = st.checkbox(genres_list[12], key = 12)
		check13 = st.checkbox(genres_list[13], key = 13)
		check14 = st.checkbox(genres_list[14], key = 14)

	with col4:
		check15 = st.checkbox(genres_list[15], key = 15)
		check16 = st.checkbox(genres_list[16], key = 16)
		check17 = st.checkbox(genres_list[17], key = 17)
		check18 = st.checkbox(genres_list[18], key = 18)
		check19 = st.checkbox(genres_list[19], key = 19)

	if check:
		input += genres_list[0] + ', '
	if check1:
		input += genres_list[1] + ', '
	if check2:
		input += genres_list[2] + ', '
	if check3:
		input += genres_list[3] + ', '
	if check4:
		input += genres_list[4] + ', '
	if check5:
		input += genres_list[5] + ', '
	if check6:
		input += genres_list[6] + ', '
	if check7:
		input += genres_list[7] + ', '
	if check8:
		input += genres_list[8] + ', '
	if check9:
		input += genres_list[9] + ', '
	if check10:
		input += genres_list[10] + ', '
	if check11:
		input += genres_list[11] + ', '
	if check12:
		input += genres_list[12] + ', '
	if check13:
		input += genres_list[13] + ', '
	if check14:
		input += genres_list[14] + ', '
	if check15:
		input += genres_list[15] + ', '
	if check16:
		input += genres_list[16] + ', '
	if check17:
		input += genres_list[17] + ', '
	if check18:
		input += genres_list[18] + ', '
	if check19:
		input += genres_list[19]

	st.write('You have selected : ', input)
	slider = st.slider("How many recommendations ?", 1, 20, 10)
	if (st.button('Get Recommendation')):
		result = recommend(input, anime['Genres'], slider)#st.write(result)#st.balloons()
		st.subheader('Check this out !')
		for row in result.iterrows():
			url_image = row[1][9]
			rec_title = row[1][2]
			rec_score = round(row[1][3],1)
			rec_genre = row[1][4]
			rec_syns = row[1][5]
			rec_ep = row[1][6]
			url_source = row[1][10]
			#col5, padding,col6, padding,col7 = st.columns((20,5,20,5,20))
			col5,col6,col7 = st.columns(3)
			with col5:
				st.image(url_image, width=200)

			with col6:
				st.subheader(rec_title)
				st.write('Score : ',str(rec_score) + ' /🔟')
				st.write('Genres 🏷️ : ',rec_genre)
				st.write('Episodes ⏳ : ', rec_ep)

				#st.write("ℹ️ More info [here](%s)" % url)
				#st.write('Quick look ▶️ 🎬 : ',)


			with col7:
				st.write('')
				st.write('Synopsis 📜 : ')
				st.write(cut_text(rec_syns))
				st.write('')
				st.write("ℹ️ More info [here](%s)" % url_source)
