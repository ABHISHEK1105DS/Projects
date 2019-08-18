#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd

# This is for making some large tweets to be displayed
pd.options.display.max_colwidth = 100


# In[12]:


train_data= pd.read_csv(r'D:\ml\tweets-sentiment-analysis/train.csv',encoding='ISO-8859-1')


# In[13]:


train_data


# In[14]:


# We will now take a look at random tweets to gain more insights
rand_indexs = np.random.randint(1,len(train_data),50).tolist()


# In[18]:


train_data["SentimentText"][rand_indexs]


# In[ ]:


"""
You will not have the same results at each execution because of the randomization. For me, after some execution, I noticed this:

There is tweets with a url (like tweet 35546): we must think about a way to handle URLs, I thought about deleting them because a domain name or the protocol used will not make someone happy or sad unless the domain name is 'food.com'.
The use of hashtags: we should keep only the words without '#' so words like python and the hashtag '#python' can be seen as the same word, and of course they are.
Words like 'as', 'to' and 'so' should be deleted, because they only serve as a way to link phrases and words

"""


# In[19]:


# emoticons=>make sure anlaysis to classify emoticon as happy and sad


# In[22]:


import re
tweets_text = train_data.SentimentText.str.cat()


# In[24]:


emos = set(re.findall(r" ([xX:;][-']?.) ",tweets_text))
emos_count = []


# In[25]:


emos


# In[26]:


for emo in emos:
    emos_count.append((tweets_text.count(emo), emo))
sorted(emos_count,reverse=True)


# In[27]:


HAPPY_EMO = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
SAD_EMO = r" (:'?[/|\(]) "
print("Happy emoticons:", set(re.findall(HAPPY_EMO, tweets_text)))
print("Sad emoticons:", set(re.findall(SAD_EMO, tweets_text)))


# In[ ]:


# Most used words 
# (What we are going to do next is to define a function that will show us top words, so we may fix things before running our learning algorithm. This function takes as input a text and output words sorted according to their frequency, starting with the most used word.)


# In[28]:


import nltk
from nltk.tokenize import word_tokenize


# In[29]:


def most_used_words(text):
    tokens = word_tokenize(text)
    frequency_dist = nltk.FreqDist(tokens)
    print("There is %d different words" % len(set(tokens)))
    return sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)


# In[30]:


most_used_words(train_data.SentimentText.str.cat())[:100]


# In[31]:


# Stop words
# (What we can see is that stop words are the most used, but in fact they don't help us determine if a tweet is happy/sad, however, they are consuming memory and they are making the learning process slower, so we really need to get rid of them.)


# In[32]:


from nltk.corpus import stopwords


# In[33]:


mw = most_used_words(train_data.SentimentText.str.cat())
most_words = []


# In[34]:


for w in mw:
    if len(most_words) == 1000:
        break
    if w in stopwords.words("english"):
        continue
    else:
        most_words.append(w)


# In[35]:


sorted(most_words)


# In[ ]:


# Stemming
# ( You should have noticed something, right? There are words that have the same meaning, but written in a different manner, sometimes in the plural and sometimes with a suffix (ing, es ...), this will make our model think that they are different words and also make our vocabulary bigger (waste of memory and time for the learning process). The solution is to reduce those words with the same root, this is called stemming. )


# In[36]:


from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


# In[37]:


def stem_tokenize(text):
    stemmer = SnowballStemmer("english")
    stemmer = WordNetLemmatizer()
    return [stemmer.lemmatize(token) for token in word_tokenize(text)]


# In[38]:


def lemmatize_tokenize(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in word_tokenize(text)]


# In[ ]:


# Prepare the data
# (Bag of Words)
# We are going to use the Bag of Words algorithm, which basically takes a text as input, extract words from it (this is our vocabulary) to use them in the vectorization process. When a tweet comes in, it will vectorize it by counting the number of occurrences of each word in our vocabulary.)


# In[39]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


# Building the pipeline
# It's always a good practice to make a pipeline of transformation for your data, it will make the process of data transformation really easy and reusable. We will implement a pipeline for transforming our tweets to something that our ML models can digest (vectors).


# In[40]:


from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline


# In[41]:


# We need to do some preprocessing of the tweets.
# We will delete useless strings (like @, # ...)


# In[42]:


class TextPreProc(BaseEstimator,TransformerMixin):
    def __init__(self, use_mention=False):
        self.use_mention = use_mention
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # We can choose between keeping the mentions
        # or deleting them
        if self.use_mention:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")
        else:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")
            
        # Keeping only the word after the #
        X = X.str.replace("#", "")
        X = X.str.replace(r"[-\.\n]", "")
        # Removing HTML garbage
        X = X.str.replace(r"&\w+;", "")
        # Removing links
        X = X.str.replace(r"https?://\S*", "")
        # replace repeated letters with only two occurences
        # heeeelllloooo => heelloo
        X = X.str.replace(r"(.)\1+", r"\1\1")
        # mark emoticons as happy or sad
        X = X.str.replace(HAPPY_EMO, " happyemoticons ")
        X = X.str.replace(SAD_EMO, " sademoticons ")
        X = X.str.lower()
        return X


# In[43]:


# This is the pipeline that will transform our tweets to something eatable.
# You can see that we are using our previously defined stemmer, it will
# take care of the stemming process.
# For stop words, we let the inverse document frequency do the job
from sklearn.model_selection import train_test_split

sentiments = train_data['Sentiment']
tweets = train_data['SentimentText']


# In[44]:


vectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize, ngram_range=(1,2))
pipeline = Pipeline([
    ('text_pre_processing', TextPreProc(use_mention=True)),
    ('vectorizer', vectorizer),
])

# Let's split our data into learning set and testing set
# This process is done to test the efficency of our model at the end.
# You shouldn't look at the test data only after choosing the final model
learn_data, test_data, sentiments_learning, sentiments_test = train_test_split(tweets, sentiments, test_size=0.3)

# This will tranform our learning data from simple text to vector
# by going through the preprocessing tranformer.
learning_data = pipeline.fit_transform(learn_data)


# In[45]:


# Select a model


# In[47]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

lr = LogisticRegression()
bnb = BernoulliNB()
mnb = MultinomialNB()

models = {
    'logitic regression': lr,
    'bernoulliNB': bnb,
    'multinomialNB': mnb,
}


# In[48]:


for model in models.keys():
    scores = cross_val_score(models[model], learning_data, sentiments_learning, scoring="f1", cv=10)
    print("===", model, "===")
    print("scores = ", scores)
    print("mean = ", scores.mean())
    print("variance = ", scores.var())
    models[model].fit(learning_data, sentiments_learning)
    print("score on the learning data (accuracy) = ", accuracy_score(models[model].predict(learning_data), sentiments_learning))
    print("")


# In[49]:


# GridSearchCV to choose the best parameters to use.


# In[50]:


# what the GridSearchCV does is trying different set of parameters, and for each one, it runs a cross validation and estimate the score. At the end we can see what are the best parameter and use them to build a better classifier.


# In[51]:


from sklearn.model_selection import GridSearchCV

grid_search_pipeline = Pipeline([
    ('text_pre_processing', TextPreProc()),
    ('vectorizer', TfidfVectorizer()),
    ('model', MultinomialNB()),
])


# In[52]:


params = [
    {
        'text_pre_processing__use_mention': [True, False],
        'vectorizer__max_features': [1000, 2000, 5000, 10000, 20000, None],
        'vectorizer__ngram_range': [(1,1), (1,2)],
    },
]
grid_search = GridSearchCV(grid_search_pipeline, params, cv=5, scoring='f1')
grid_search.fit(learn_data, sentiments_learning)
print(grid_search.best_params_)


# In[53]:


mnb.fit(learning_data, sentiments_learning)


# In[54]:


testing_data = pipeline.transform(test_data)
mnb.score(testing_data, sentiments_test)


# In[56]:


sub_data= pd.read_csv(r'D:\ml\tweets-sentiment-analysis/test.csv',encoding='ISO-8859-1')
sub_learning = pipeline.transform(sub_data.SentimentText)
sub = pd.DataFrame(sub_data.ItemID, columns=("ItemID", "Sentiment"))
sub["Sentiment"] = mnb.predict(sub_learning)
print(sub)


# In[ ]:





# In[ ]:


model = MultinomialNB()
model.fit(learning_data, sentiments_learning)
tweet = pd.Series([input(),])
tweet = pipeline.transform(tweet)
proba = model.predict_proba(tweet)[0]
print("The probability that this tweet is sad is:", proba[0])
print("The probability that this tweet is happy is:", proba[1])dsa


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




