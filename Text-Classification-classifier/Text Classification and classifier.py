#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
The goal with text classification can be pretty broad. Maybe we're trying to classify text as about politics or the military. Maybe we're trying to classify it by the gender of the author who wrote it. A fairly popular text classification task is to identify a body of text as either spam or not spam, for things like email filters. In our case, we're going to try to create a sentiment analysis algorithm.

To do this, we're going to start by trying to use the movie reviews database that is part of the NLTK corpus. From there we'll try to use words as "features" which are a part of either a positive or negative movie review. The NLTK corpus movie_reviews data set has the reviews, and they are labeled already as positive or negative. This means we can train and test with this data. First, let's wrangle our data.


"""


# In[6]:


import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
# document for training
print(documents[1])
"""
Basically, in plain English, the above code is translated to: In each category (we have pos or neg), take all of the file IDs (each review has its own ID), then store the word_tokenized version (a list of words) for the file ID, followed by the positive or negative label in one big list.

Next, we use random to shuffle our documents. This is because we're going to be training and testing. If we left them in order, chances are we'd train on all of the negatives, some positives, and then test only against positives. We don't want that, so we shuffle the data."""


# In[ ]:


"""

Next, we use random to shuffle our documents. This is because we're going to be training and testing. If we left them in order, chances are we'd train on all of the negatives, some positives, and then test only against positives. We don't want that, so we shuffle the data."""
"""


# In[7]:


# common word used most time
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
# list to freqcy distribution most common word to least common
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))


# In[8]:


# stupid appear 253 time
print(all_words["stupid"])


# In[11]:



word_features = list(all_words.keys())[:3000]
"""
Mostly the same as before, only with now a new variable, word_features, which contains the top 3,000 most common words. Next, we're going to build a quick function that will find these top 3,000 words in our positive and negative documents, marking their presence as either positive or negative:

"""


# In[13]:


def find_features(document):
# one basically iteration mean set from document mean unique    
    words = set(document)        
    features = {}
    for w in word_features:
        features[w] = (w in words)
#         win words boolean value top 3000 words in document true or false

    return features


# In[14]:


print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))


# In[15]:


featuresets = [(find_features(rev), category) for (rev, category) in documents]


# In[21]:


"""

Now it is time to choose an algorithm, separate our data into training and testing sets, and press go! The algorithm that we're going to use first is the Naive Bayes classifier. This is a pretty popular algorithm used in text classification, so it is only fitting that we try it out first. Before we can train and test our algorithm, however, we need to go ahead and split up the data into a training set and a testing set.

You could train and test on the same dataset, but this would present you with some serious bias issues, so you should never train and test against the exact same data. To do this, since we've shuffled our data set, we'll assign the first 1,900 shuffled reviews, consisting of both positive and negative reviews, as the training set. Then, we can test against the last 100 to see how accurate we are.

"""


# In[24]:


print(featuresets[190])


# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


# set that we'll train our classifier with first 1900 wordfs
training_set = featuresets[:1900]

# set that we'll test against.after 1900 words
testing_set = featuresets[1900:]


# In[18]:


classifier = nltk.NaiveBayesClassifier.train(training_set)


# In[19]:


print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)


# In[20]:


classifier.show_most_informative_features(15)


# In[ ]:


"""

What this tells you is the ratio of occurences in negative to positive, or visa versa, for every word. So here, we can see that the term "insulting" appears 10.6 more times as often in negative reviews as it does in positive reviews. Ludicrous, 10.1.

Now, let's say you were totally content with your results, and you wanted to move forward, maybe using this classifier to predict things right now. It would be very impractical to train the classifier, and retrain it every time you needed to use it. As such, you can save the classifier using the pickle module. Let's do that next.
"""


# In[ ]:


# pickle
"""
Training classifiers and machine learning algorithms can take a very long time, especially if you're training against a larger data set. Ours is actually pretty small. Can you imagine having to train the classifier every time you wanted to fire it up and use it? What horror! Instead, what we can do is use the Pickle module to go ahead and serialize our classifier object, so that all we need to do is load that file in real quick.

So, how do we do this? The first step is to save the object. To do this, first you need to import pickle at the top of your script, then, after you have trained with .train() the classifier, you can then call the following lines:

"""


# 

# In[25]:



import pickle


# In[26]:


# write in byte=wb
save_classifier = open("naivebayes.pickle","wb")
# what we want to dump and where want to dump
pickle.dump(classifier, save_classifier)
save_classifier.close()


# In[27]:


"""

This opens up a pickle file, preparing to write in bytes some data. Then, we use pickle.dump() to dump the data. The first parameter to pickle.dump() is what are you dumping, the second parameter is where are you dumping it.

After that, we close the file as we're supposed to, and that is that, we now have a pickled, or serialized, object saved in our script's directory!

Next, how would we go about opening and using this classifier? The .pickle file is a serialized object, all we need to do now is read it into memory, which will be about as quick as reading any other ordinary file. To do this:
"""


# In[30]:


classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


# In[31]:


print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)


# In[32]:


classifier.show_most_informative_features(15)


# In[ ]:





# In[29]:


"""
Here, we do a very similar process. We open the file to read as bytes. Then, we use pickle.load() to load the file, and we save the data to the classifier variable. Then we close the file, and that is that. We now have the same classifier object as before!

Now, we can use this object, and we no longer need to train our classifier every time we wanted to use it to classify.

While this is all fine and dandy, we're probably not too content with the 60-75% accuracy we're getting. What about other classifiers? Turns out, there are many classifiers, but we need the scikit-learn (sklearn) module. Luckily for us, the people at NLTK recognized the value of incorporating the sklearn module into NLTK, and they have built us a little API to do it. That's what we'll be doing in the next tutorial.

"""


# In[33]:


"""
We've seen by now how easy it can be to use classifiers out of the box, and now we want to try some more! The best module for Python to do this with is the Scikit-learn (sklearn) module.

If you would like to learn more about the Scikit-learn Module, I have some tutorials on machine learning with Scikit-Learn.


 
Luckily for us, the people behind NLTK forsaw the value of incorporating the sklearn module into the NLTK classifier methodology. As such, they created the SklearnClassifier API of sorts. To use that, you just need to import it like:
"""


# In[34]:


from nltk.classify.scikitlearn import SklearnClassifier


# In[35]:


from sklearn.naive_bayes import MultinomialNB,BernoulliNB


# In[36]:


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))


# In[37]:


from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


# In[38]:


print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


# In[ ]:


"""

In this tutorial, we discuss a few issues. The most major issue is that we have a fairly biased algorithm. You can test this yourself by commenting-out the shuffling of the documents, then training against the first 1900, and leaving the last 100 (all positive) reviews. Test, and you will find you have very poor accuracy.

Conversely, you can test against the first 100 data sets, all negative, and train against the following 1900. You will find very high accuracy here. This is a bad sign. It could mean a lot of things, and there are many options for us to fix it.


 
That said, the project I have in mind for us suggests we go ahead and use a different data set anyways, so we will do that. In the end, we will find this new data set still contains some bias, and that is that it picks up negative things more often. The reason for this is that negative reviews tend to be "more negative" than positive reviews are positive. Handling this can be done with some simple weighting, but it can also get complex fast. Maybe a tutorial for another day. For now, we're going to just grab a new dataset, which we'll be discussing in the next tutorial.


"""

