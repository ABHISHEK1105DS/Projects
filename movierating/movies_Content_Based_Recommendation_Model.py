#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Reading ratings file
ratings = pd.read_csv(r'D:\ml\movierating/ratings.csv', sep='\t')
# Reading users file
users = pd.read_csv(r'D:\ml\movierating/users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
movies = pd.read_csv(r'D:\ml\movierating/movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])


# In[3]:


ratings.head()


# In[4]:


# Reading ratings file
# Ignore the timestamp column,user_emb_id,movie_emb_id and Unnamed
ratings = pd.read_csv(r'D:\ml\movierating/ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])


# In[5]:


ratings.head()


# In[6]:


ratings.info()


# In[7]:


users.head()


# In[8]:


users.info()


# In[9]:


movies.head()


# In[10]:


movies.info()


# In[11]:


# Data Exploration


# In[12]:


# Titles
# Are there certain words that feature more often in Movie Titles? I'll attempt to figure this out using a word-cloud visualization.


# In[13]:


# Import new libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import wordcloud
from wordcloud import WordCloud, STOPWORDS


# In[14]:


# Create a wordcloud of the movie titles
movies['title'] = movies['title'].fillna("").astype('str')
title_corpus = ' '.join(movies['title'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)


# In[15]:


# Plot the wordcloud
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()


# In[16]:


# rating
ratings['rating'].describe()


# In[17]:


# Import seaborn library
import seaborn as sns
sns.set_style('whitegrid')
sns.set(font_scale=1.5)
get_ipython().run_line_magic('matplotlib', 'inline')

# Display distribution of rating
sns.distplot(ratings['rating'].fillna(ratings['rating'].median()))


# In[18]:


# Import seaborn library
import seaborn as sns
sns.set_style('whitegrid')
sns.set(font_scale=1.5)
get_ipython().run_line_magic('matplotlib', 'inline')

# Display distribution of rating
sns.distplot(ratings['rating'].fillna(ratings['rating'].mean()))


# In[19]:


# Join all 3 files into one dataframe
dataset = pd.merge(pd.merge(movies, ratings),users)
# Display 20 movies with highest ratings
dataset[['title','genres','rating']].sort_values('rating', ascending=False).head(20)


# In[20]:


# genres
# The genres variable will surely be important while building the recommendation engines since it describes the content of the film (i.e. Animation, Horror, Sci-Fi). A basic assumption is that films in the same genre should have similar contents. I'll attempt to see exactly which genres are the most popular.


# In[21]:


# Make a census of the genre keywords
genre_labels = set()
for s in movies['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))


# In[22]:


# Function that counts the number of times each of the genre keywords appear
def count_word(dataset, ref_col, census):
    """dataset=>movies,ref_col=genres row in movie,census=genres label"""
    keyword_count = dict()
    for s in census: 
        keyword_count[s] = 0
    for census_keywords in dataset[ref_col].str.split('|'):        
        if type(census_keywords) == float and pd.isnull(census_keywords): 
            continue        
        for s in [s for s in census_keywords if s in census]:
            """2 for loop with if statement"""
            if pd.notnull(s): 
                keyword_count[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences

# Calling this function gives access to a list of genre keywords which are sorted by decreasing frequency
keyword_occurences= count_word(movies, 'genres', genre_labels)
keyword_occurences


# In[ ]:





# In[23]:


# Define the dictionary used to produce the genre wordcloud
genres = dict()
trunc_occurences = keyword_occurences[0:18]


# In[24]:


for s in trunc_occurences:
    
    genres[s[0]] = s[1]


# In[25]:


# Create the wordcloud
genre_wordcloud = WordCloud(width=1000,height=400, background_color='white')
genre_wordcloud.generate_from_frequencies(genres)

# Plot the wordcloud
f, ax = plt.subplots(figsize=(16, 8))
plt.imshow(genre_wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[30]:


# Content-Based Recommendation Model
# """With all that theory in mind, I am going to build a Content-Based Recommendation Engine that computes similarity between movies based on movie genres. It will suggest movies that are most similar to a particular movie based on its genre. To do so, I will make use of the file movies.csv."""


# In[27]:


# Break up the big genre string into a string array
movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')


# In[28]:


movies['genres']


# In[31]:


#  do not have a quantitative metric to judge our machine's performance so this will have to be done qualitatively. In order to do so, I'll use TfidfVectorizer function from scikit-learn, which transforms text to feature vectors that can be used as input to estimator.
# to convert text into integer
# """convert text to word count vectors with CountVectorizer.
# to convert text to word frequency vectors with TfidfVectorizer.to convert text to unique integers with HashingVectorizer."""


# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape


# In[ ]:


# I will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two movies. Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score. Therefore, we will use sklearn's linear_kernel instead of cosine_similarities since it is much faster.


# In[33]:


from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[:4, :4]


# In[34]:


# I now have a pairwise cosine similarity matrix for all the movies in the dataset. The next step is to write a function that returns the 20 most similar movies based on the cosine similarity score.


# In[36]:


# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])


# In[37]:


indices


# In[38]:


# Function that get movie recommendations based on the cosine similarity score of movie genres


# In[39]:


def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[40]:


genre_recommendations('Good Will Hunting (1997)').head(20)


# In[44]:


genre_recommendations('Freejack (1992)').head(20)


# In[ ]:


"""Disadvantages
Finding the appropriate features is hard.
Does not recommend items outside a user's content profile.
Unable to exploit quality judgments of other users.

"""


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


genres


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




