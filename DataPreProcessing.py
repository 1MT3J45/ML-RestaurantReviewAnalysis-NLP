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
# -------------------------------------------- DATASET = fullA
dataset = fullB
# Creating a Corpus
corpus = []
for i in range(0, dataset.__len__()):
    review = re.sub("[^a-zA-Z]", ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()     # TODO FIND POINT OF DIFFERENCE wrt LEMMATIZATION
    nl = WordNetLemmatizer() # TODO Apply Lemmatization after POS Tagging

    review = [ps.stem(nl.lemmatize(word, pos='v')) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag Of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2844)

RestoReview = pd.DataFrame()
# POS Tagging
for i in range(0, len(corpus)):
    sent = nltk.word_tokenize(corpus[i])
    PoS_tags = nltk.pos_tag(sent)

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    one_sentence = corpus[i]
    scores = sia.polarity_scores(text=one_sentence)
    # print("POS:", scores.get('pos'))
    # print("NEG:", scores.get('neg'))
    # print("NEU:", scores.get('neu'))

    POS = scores.get('pos')
    NEG = scores.get('neg')
    NEU = scores.get('neu')
    RES = str()

    if POS > NEG:
        RES = 1
    elif NEG > POS:
        RES = 0
    elif NEU >= 0.5 or POS > NEU:
        RES = 1
    elif NEU < 0.5:
        RES = 0

    # print(RES)

    j = int((i+1)/len(corpus) *100)
    sys.stdout.write(('=' * j) + ('' * (100 - j)) + ("\r [ %d" % j + "% ] "))
    sys.stdout.flush()

    RestoReview = RestoReview.append({'reviews': corpus[i], 'polarity': int(RES)}, ignore_index=True)
    RestoReview.to_csv("RestoReview.csv", index=False)
    RestoReview.head(10)
print("\n")
try:
    os.system("libreoffice --calc RestoReview.csv")
except:
    print("This feature works with Ubuntu OS with LibreOffice only!")

from nltk import bigrams

# for w1, tag1, w2 , tag2 in nltk.bigrams(PoS_tags):
#     if tag1.startswith('JJ'or 'RB')

# DATAFRAME OP - Feature Extraction
# text/review , lemmas , bitagged feature , dep. feature

# DATAFRAME OP - Unique Feature Extraction
# text/review , lemmas , bitagged feature , dep. feature (Unique from lemmas & bitagged feature)

# ASPECT CAT DATAFRAME OP - Aspect category grouping
# Aspect cat (y), Wordnet (SYNONYMS) , Conceptnet (CONCEPTS) , COMBINED FEATURE SET (X)
