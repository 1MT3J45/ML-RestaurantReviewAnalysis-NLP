import os
import pandas as pd
import nltk
import sys
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

testB = pd.read_csv("CSV/Restaurants_Test_Data_phaseB.csv")
trainB = pd.read_csv("CSV/Restaurants_Train_v2.csv")

print(testB.head(5))
print(trainB.head(5))

trainB_1 = trainB.iloc[:, [0, 7, 5]]
testB_1 = testB.iloc[:, [0, 5, 4]]

fullB = pd.concat([testB_1, trainB_1], axis=0, ignore_index=True)

nltk.download('stopwords')
import re

dataset = fullB     # MAJOR DATASET
corpus = []         # CORPUS (will collect required data)
# --------------------- STREAMS ----------------------------

# ---------------------------------------------------------- STREAM 1 - LEMMATIZATION
for i in range(0, dataset.__len__()):
    review = re.sub("[^a-zA-Z]", ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()

    # Learn More at https://www.quora.com/What-is-difference-between-stemming-and-lemmatization
    ps = PorterStemmer()
    nl = WordNetLemmatizer()

    review = [ps.stem(nl.lemmatize(word, pos='v')) for word in review if not word in set(stopwords.words('english'))]
    #review = ' '.join(review)
    review = list(set(review))
    corpus.append(review)

# ----------------------------------------------------------- STREAM 2 - BIGRAMS
PoS_Tag_sent = list()
s2_corpus = []
for i in range(dataset.__len__()):
    sent = nltk.word_tokenize(dataset.iloc[i, 0])
    PoS_Tag_sent = nltk.pos_tag(sent)

    for (w1, tag1), (w2, tag2) in nltk.bigrams(PoS_Tag_sent):
        if tag1.startswith('JJ') and tag2.startswith('NN'):     # R1
            s2_corpus.append((w1 + ' ' + w2))
        elif tag1.startswith('RB') and tag2.startswith('JJ'):   # R2
            s2_corpus.append((w1 + ' ' + w2))
        elif tag1.startswith('JJ') and tag2.startswith('JJ'):   # R3
            s2_corpus.append((w1 + ' ' + w2))
        elif tag1.startswith('NN') and tag2.startswith('JJ'):   # R4
            s2_corpus.append((w1 + ' ' + w2))
        elif tag1.startswith('RB') and tag2.startswith('VB'):   # R5
            s2_corpus.append((w1 + ' ' + w2))
        elif tag1.startswith('VB') and tag2.startswith('NN'):   # R6
            s2_corpus.append((w1 + ' ' + w2))
        elif tag1.startswith('JJ') and tag2.startswith('VB'):   # R7
            s2_corpus.append((w1 + ' ' + w2))
        elif tag1.startswith('RB') and tag2.startswith('RB'):   # R8
            s2_corpus.append((w1 + ' ' + w2))
        elif tag1.startswith('RB') and tag2.startswith('VB'):   # R9
            s2_corpus.append((w1 + ' ' + w2))

    print(s2_corpus)
# ----------------------------------------------------------- STREAM 3 - DEPENDENCY FEATURES
# TODO Associate All JARs with Python Code

