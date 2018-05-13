from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from multiprocessing.dummy import Pool as ThreadPool

import os
import pandas as pd
import nltk
import numpy as np

import re
import spacy
from nltk.corpus import wordnet
from autocorrect import spell

import progressbar as bar
import extractUnique as xq
import tristream_processor as stream
import pickle

testB = pd.read_csv("CSV/Restaurants_Test_Data_phaseB.csv")
trainB = pd.read_csv("CSV/Restaurants_Train_v2.csv")

# print(testB.head(5))
# print(trainB.head(5))

trainB_1 = trainB.iloc[:, [0, 7, 5]]
testB_1 = testB.iloc[:, [0, 5, 4]]

fullB = pd.concat([trainB_1], axis=0, ignore_index=True)

# nltk.download('stopwords')


dataset = fullB  # MAJOR DATA-SET


# --------------------- FUNCTIONS --------------------------


def check_dep_parse(token_dep):
    dep_str = token_dep
    check_list = list()
    if dep_str.startswith('nsub'):
        pass
    elif dep_str.startswith('amod'):
        pass
    elif dep_str.startswith('rcmod'):
        pass
    elif dep_str.startswith('dobj'):
        pass
    elif dep_str.startswith('neg'):
        pass
    else:
        return False
    return True


# --------------------- STREAM INITIALIZER ----------------------------
PoS_Tag_sent = list()
S1_corpus = []  # CORPUS (For Collecting Lemmas)
corpora = ''  # CORPORA (For Collecting Corpora of single sentence)
S2_super_corpus = []  # CORPUS (For Collecting Bigrams sentence wise)

# --------------------- SPACY SPECS ------------------------
nlp_en = spacy.load('en_core_web_sm')
plot_nlp = 0  # For Plotting of Dependency chart
S3_dep_corpus = []  # CORPUS (For Collecting Dependency Relations)

# ---------------------------------------------------------- STREAM 1 - LEMMATIZATION
try:
    for i in range(0, len(dataset)):
        review = re.sub("[^a-zA-Z]", ' ', dataset['text'][i])
        review = review.lower()
        review = review.split()

        # Learn More at https://www.quora.com/What-is-difference-between-stemming-and-lemmatization
        ps = PorterStemmer()
        nl = WordNetLemmatizer()

        review = [ps.stem(nl.lemmatize(word, pos='v')) for word in review if
                  not word in set(stopwords.words('english'))]
        review = list(set(review))
        S1_corpus.append(review)
        bar.load(i, base=dataset, text='Stream 1')
    print('Stream 1: Processed')
    # print(S1_corpus)
    # ----------------------------------------------------------- STREAM 2 - BIGRAMS

    for i in range(len(dataset)):
        sent = nltk.word_tokenize(dataset.iloc[i, 0].lower())
        PoS_Tag_sent = nltk.pos_tag(sent)

        for (w1, tag1), (w2, tag2) in nltk.bigrams(PoS_Tag_sent):
            if tag1.startswith('JJ') and tag2.startswith('NN'):  # R1
                corpora += w1 + ' ' + w2 + ';'
            elif tag1.startswith('RB') and tag2.startswith('JJ'):  # R2
                corpora += w1 + ' ' + w2 + ';'
            elif tag1.startswith('JJ') and tag2.startswith('JJ'):  # R3
                corpora += w1 + ' ' + w2 + ';'
            elif tag1.startswith('NN') and tag2.startswith('JJ'):  # R4
                corpora += w1 + ' ' + w2 + ';'
            elif tag1.startswith('RB') and tag2.startswith('VB'):  # R5
                corpora += w1 + ' ' + w2 + ';'
            elif tag1.startswith('VB') and tag2.startswith('NN'):  # R6
                corpora += w1 + ' ' + w2 + ';'
            elif tag1.startswith('JJ') and tag2.startswith('VB'):  # R7
                corpora += w1 + ' ' + w2 + ';'
            elif tag1.startswith('RB') and tag2.startswith('RB'):  # R8
                corpora += w1 + ' ' + w2 + ';'
            elif tag1.startswith('RB') and tag2.startswith('VB'):  # R9
                corpora += w1 + ' ' + w2 + ';'

        S2_super_corpus.append(corpora)
        corpora = ''
        bar.load(i, base=dataset, text='Stream 2')
    print('Stream 2: Processed')
except KeyboardInterrupt:
    print("[STAGE 1] Terminating. Human Intervention Not Allowed")
    exit(0)
except AttributeError as e:
    print("[STAGE 1] Terminating. Human Intervention Not Allowed")
    exit(0)
# ----------------------------------------------------------- STREAM 3 - DEPENDENCY FEATURES (spaCy)

try:
    for increment in range(len(dataset)):
        sentence = dataset.iloc[increment, 0].lower()
        # print(increment)
        for token in nlp_en(sentence):
            dep = check_dep_parse(token.dep_)
            if dep is True:
                # print(token.dep_, end="> ")
                # print(token.head, token)
                corpora += str(token) + ' ' + str(token.head) + ';'
            else:
                pass
        S3_dep_corpus.append(corpora)
        corpora = ''
        bar.load(increment, base=dataset, text='Stream 3')
    print('Stream 3: Processed')
    plot_nlp = nlp_en(sentence)

except TypeError as e:
    print("[STAGE 2] Unexpected Termination:", e)
    exit(0)
except KeyboardInterrupt:
    print("[STAGE 2] Human Interrupt Received! Exiting...")
    exit(0)

stream1 = pd.Series(S1_corpus)
stream2 = pd.Series(S2_super_corpus)
stream3 = pd.Series(S3_dep_corpus)
df = pd.concat([stream1, stream2, stream3], axis=1)
df = df.rename(columns={0: 'lemmas', 1: 'bigrams', 2: 'depenrel'})
df.to_csv('FeatureSet.csv', index=False)
df = pd.read_csv('FeatureSet.csv', sep=',')
# try:
#     pool = ThreadPool(2)
#     pool.map(os.system('firefox localhost:5000 &'), spacy.displacy.serve(plot_nlp, style='dep')).join()
#     exit(0)
# except OSError:
#     print("Browser must start with Graph. If doesn't please make sure to use Ubuntu with Firefox")
# except TypeError:
#     print("Browser must start with Graph. If doesn't please make sure to use Ubuntu with Firefox")

# Get Unique Features from Bi-grams, Dependency Rel
whole_df = pd.concat([dataset.iloc[0:, 0], stream1, stream2, stream3, dataset.iloc[0:, 2]], axis=1)
whole_df = whole_df.rename(columns={'text': 'reviews', 0: 'lemmas', 1: 'bigrams', 2: 'depenrel',
                                    'aspectCategories/aspectCategory/0/_category': 'aspectCategory'})
whole_df.to_csv('WholeSet.csv', index=False)
whole_df = pd.read_csv('WholeSet.csv', sep=',')
u_feat = list()
try:
    u_feat = xq.unique(whole_df=whole_df, bigram_col=2, dep_rel_col=3)
    print("Unique Features Extracted")
except KeyboardInterrupt:
    print("[STAGE 3] Manual Interrupt to Unique Features")
    exit(0)
except Exception as e:
    print('[STAGE 3] Improper Termination due to:', e)
    exit(0)
# DF with Review, Lemmas, U_feat, Aspect Cat
Feature_df = whole_df[['reviews', 'lemmas']][0:]
Feature_df = pd.concat([Feature_df, pd.Series(u_feat), whole_df.iloc[0:, -1]], axis=1)
Feature_df = Feature_df.rename(columns={0: 'ufeat'})
Feature_df.to_csv('Feature.csv', index=False)

# Aspect Cat, Lemmas + U_feat (from All sentences)
c_list = list()
try:
    c_list = xq.combiner(Feature_df=Feature_df, lemma_col=1, uniqueFeat_col=2, use_ast=True)
except KeyboardInterrupt:
    print("[STAGE 4] Manual Interrupt to Combiner")
    exit(0)
except Exception as e:
    print("[STAGE 4] Improper Termination due to:", e)
    exit(0)
ngram_list = list()
import time
import os
try:
    st = time.time()
    ngram_list = xq.get_correct_spell(word_list=c_list, split_by=';')
    et = time.time()
    if len(ngram_list) > 0:
        os.system('python2 notifyme.py 9765838775 Sir, Execution is complete!')
    else:
        os.system('python2 notifyme.py 9765838775 Sir, you might wanna check the code you wrote')
    print('Time elapsed %.3f' % float(((et-st)/60)/60))
except ValueError:
    print("[STAGE 5] Spell Checker | Interrupted")
except TypeError:
    print("[STAGE 5] Spell Checker | Multi-threading issue")
except AttributeError:
    print("[STAGE 5] Spell Checker | Attrition")
except KeyboardInterrupt:
    print("[STAGE 5] Spell Checker | Forced Drop")

# Creating Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=12488, ngram_range=(1, 2))
X = cv.fit_transform(ngram_list).toarray()
itemDict = cv.vocabulary_

# key_Book = pd.DataFrame(itemDict, index=range(itemDict.__len__()))
# key_Book.to_csv('key_Book.csv', index=True, sep=',')
# ============================== Preparing Train set =============================
# ML with Bag of Words to Aspect Categories
X_train = cv.fit_transform(ngram_list).toarray()
y_train = Feature_df['aspectCategory']

# ============================== Preparing Test set ===============================
reviews = list()
for x in range(len(testB_1)):
    l = testB_1['text'][x]
    reviews.append(l.lower())

stream1 = stream.lemmatize(testB_1)     # Lemmas
stream2 = stream.bigram(testB_1)        # Bi-grams
stream3 = stream.dep_rel(testB_1)       # Dependency Relations
testB_1 = testB_1.rename(columns={'aspectCategories/aspectCategory/0/_category': 'aspectCategory'})

correct_reviews = xq.get_correct_spell(stream3, split_by=';')
test_df = pd.concat((testB_1['text'][0:], stream1, stream2, stream3, testB_1['aspectCategory']), axis=1)
unique_feat = xq.unique(whole_df=test_df, bigram_col=2, dep_rel_col=3)
test_df = pd.concat((testB_1['text'][0:], stream1, pd.Series(unique_feat), testB_1['aspectCategory']), axis=1)
unique_list = xq.combiner(Feature_df=test_df, lemma_col=1, uniqueFeat_col=2, use_ast=False)
ngram_test = xq.get_correct_spell(word_list=unique_list, split_by=';')

cv2 = CountVectorizer(max_features=12488, ngram_range=(1, 2))
X_test = cv2.fit_transform(ngram_test).toarray()
y_test = test_df['aspectCategory']

# ----------------- PREPARING THE MACHINE --------------------------

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=11)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_score
print(confusion_matrix(y_test, y_pred))
prf = precision_recall_fscore_support(y_test, y_pred)
li = ['Precision', 'Recall\t', 'F1 Measure']
for i in range(len(prf)-1):
    x = prf[i]*100.0
    print('%s \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f' % (li[i], x[0], x[1], x[2], x[3], x[4]))
