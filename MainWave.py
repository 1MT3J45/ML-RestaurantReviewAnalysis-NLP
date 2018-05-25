from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from multiprocessing.dummy import Pool as ThreadPool

import os
import time
import pandas as pd
import nltk
import numpy as np

import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer

import progressbar as bar
import extractUnique as xq
import tristream_processor as stream

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

_start = time.time()

testB = pd.read_csv("CSV/Restaurants_Test_Data_phaseB.csv")
trainB = pd.read_csv("CSV/Restaurants_Train_v2.csv")

trainB_1 = trainB.iloc[:, [0, 7, 5]]
testB_1 = testB.iloc[:, [0, 5, 4]]
del testB

fullB = pd.concat([trainB_1], axis=0, ignore_index=True)

dataset = fullB  # MAJOR DATA-SET

# --------------------- FUNCTIONS --------------------------


def check_dep_parse(token_dep):
    dep_str = token_dep
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


def streamers(train_dataset):
    dataset = train_dataset
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
        del PoS_Tag_sent, tag1, tag2, w1, w2, sent
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
    del sentence, dep, increment, token

    stream1 = pd.Series(S1_corpus)
    stream2 = pd.Series(S2_super_corpus)
    stream3 = pd.Series(S3_dep_corpus)

    stream1.to_csv('stream1.csv', index=False); stream2.to_csv('stream2.csv', index=False)
    stream3.to_csv('stream3.csv', index=False)

    del S1_corpus, S2_super_corpus, S3_dep_corpus

    return stream1, stream2, stream3


def sheet_generator(s1, s2, s3):
    stream1 = s1
    stream2 = s2
    stream3 = s3

    df = pd.concat([stream1, stream2, stream3], axis=1)
    df = df.rename(columns={0: 'lemmas', 1: 'bigrams', 2: 'depenrel'})
    df.to_csv('FeatureSet.csv', index=False)
    df = pd.read_csv('FeatureSet.csv', sep=',')

    del df
    # try:
    #     pool = ThreadPool(2)
    #     pool.map(os.system('firefox localhost:5000 &'), spacy.displacy.serve(plot_nlp, style='dep')).join()
    #     exit(0)
    # except OSError:
    #     print("Browser must start with Graph. If doesn't please make sure to use Ubuntu with Firefox")
    # except TypeError:
    #     print("Browser must start with Graph. If doesn't please make sure to use Ubuntu with Firefox")

    # Get Unique Features from Bi-grams, Dependency Rel
    stream2 = stream2.rename(1)
    stream3 = stream3.rename(2)
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
    del whole_df,

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
    return Feature_df, c_list


def corrector(combined_features_list):
    c_list = combined_features_list
    ngram_list = list()

    try:
        st = time.time()
        ngram_list = xq.get_correct_spell(word_list=c_list, split_by=';')
        et = time.time()
        print('Time elapsed %.3f' % float(((et-st)/60)/60))
    except ValueError:
        print("[STAGE 5] Spell Checker | Interrupted")
    except TypeError:
        print("[STAGE 5] Spell Checker | Multi-threading issue")
    except AttributeError:
        print("[STAGE 5] Spell Checker | Attrition")
    except KeyboardInterrupt:
        print("[STAGE 5] Spell Checker | Forced Drop")

    pd.Series(ngram_list).to_csv('ngram_list.csv', index=False)
    return ngram_list

# Creating Bag of Words Model
def creating_bow(corrected_list, features_dataframe, max_features=33433):
    ngram_list = list(corrected_list)
    Feature_df = features_dataframe
    max_ft = max_features

    cv = CountVectorizer(max_features=max_ft, ngram_range=(1, 2))
    # key_Book = pd.DataFrame(itemDict, index=range(itemDict.__len__()))
    # key_Book.to_csv('key_Book.csv', index=True, sep=',')
    # ============================== Preparing Train set =============================
    # ML with Bag of Words to Aspect Categories
    X_train = cv.fit_transform(ngram_list).toarray()
    y_train = Feature_df['aspectCategory']
    del ngram_list
    return X_train, y_train


# ============================== Preparing Test set ===============================
def streamers_test(main_dataset):
    testB_1 = main_dataset
    reviews = list()
    for x in range(len(testB_1)):
        l = testB_1['text'][x]
        reviews.append(l.lower())

    stream1 = stream.lemmatize(testB_1)     # Lemmas
    stream2 = stream.bigram(testB_1)        # Bi-grams
    stream3 = stream.dep_rel(testB_1)       # Dependency Relations
    testB_1 = testB_1.rename(columns={'aspectCategories/aspectCategory/0/_category': 'aspectCategory'})
    return stream1, stream2, stream3, testB_1


def sheet_generator_test(stream1_test, stream2_test, stream3_test, dataset_test):
    stream1 = stream1_test
    stream2 = stream2_test
    stream3 = stream3_test
    testB_1 = dataset_test

    test_df = pd.concat((testB_1['text'][0:], stream1, stream2, stream3, testB_1['aspectCategory']), axis=1)
    unique_feat = xq.unique(whole_df=test_df, bigram_col=2, dep_rel_col=3)

    test_df = pd.concat((testB_1['text'][0:], stream1, pd.Series(unique_feat), testB_1['aspectCategory']), axis=1)
    unique_list = xq.combiner(Feature_df=test_df, lemma_col=1, uniqueFeat_col=2, use_ast=False)

    ngram_test = list()
    arr = os.listdir('.')
    if 'ngram_test.csv' not in arr:
        ngram_test = xq.get_correct_spell(word_list=unique_list, split_by=';')
        pd.Series(ngram_test).to_csv('ngram_test.csv', index=False)
        return ngram_test, test_df
    else:
        ngram_test = pd.read_csv('ngram_test.csv', header=None)
        print('\nAVAILABLE CORRECTED NGRAMS')
        return list(ngram_test[0]), test_df


def creating_bow_test(corrected_list_test, dataframe_test, max_features=33433):
    ngram_test = corrected_list_test
    test_df = dataframe_test
    max_ft = max_features

    cv2 = CountVectorizer(max_features=max_ft, ngram_range=(1, 2))
    X_test = cv2.fit_transform(ngram_test).toarray()
    y_test = test_df['aspectCategory']
    return X_test, y_test


# ----------------- PREPARING THE MACHINE --------------------------
def the_machine(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=11)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    prf = precision_recall_fscore_support(y_test, y_pred)
    li2 = list(rfc.classes_)
    li = ['Precision', 'Recall\t', 'F1 Measure']
    print('\t\t %s \t %.8s \t %s \t %s \t %s' % (li2[0], li2[1], li2[2], li2[3], li2[4]))
    for i in range(len(prf)-1):
        x = prf[i]*100.0
        print('%s \t %.2f \t\t %.2f \t %.2f \t %.2f \t %.2f' % (li[i], x[0], x[1], x[2], x[3], x[4]))


def executor():
    '''
    streamers: Will create all 3 streams of Lemmas, Bi-grams and Dependency Relations for Training Set
    sheet_generator: Does the Job of combining Features from 3 Streams to One Unique set of Feature for TrS
    corrector: Corrects all the ngrams and produces a list of uni-grams and bi-grams from TrS
    creating_bow: Creates Bag of Words from Corrected ngrams of TrS
    streamers_test: Will create all 3 streams of Lemmas, Bi-grams and Dependency Relations for Test Set
    sheet_generator_test: Does the Job of combining Features from 3 Streams to One Uniquely
                        corrected set of Feature for TeS
    creating_bow_test: Creates Bag of Words from Corrected ngrams of TeS

    ARGUMENTS
    train_ds: Dataset
    :return:
    '''
    import fnmatch as match
    all_streams = list()
    a = b = c = pd.Series()
    max_feat = int(input('Enter Max no. of features:'))

    while True:
        global fullB, testB_1

        choice = int(input("""\t\t\t\tMENU\n
-------- FOR TRAIN SET ---------(1 Hr 20 Mins)
1. Perform Lemmatization, Bi-grams formation \n\t\t& Dependency Relations\n
2. Combine into Unique Features (4Secs)\n
3. Create Bag of Words Model (2Secs)\n
-------- FOR TEST SET ----------(50 Mins)
4. Perform Pre-processing & Processing on Test Set 
-------- MACHINE LEARNING ------
5. Call Machine
6. Exit
\t Choice:"""))
        if choice == 1:
            arr = os.listdir('.')
            exists = [item.startswith('stream*') for item in arr if item.startswith('stream')]
            if 'False' in exists:
                a, b, c = streamers(fullB)
                all_streams.append(a)
                all_streams.append(b)
                all_streams.append(c)
            else:
                print('ALREADY PROCESSED: GO TO STEP 2')

        elif choice == 2:
            arr = os.listdir('.')
            exists = [item.startswith('stream*') for item in arr if item.startswith('stream')]
            if 'False' in exists:
                print('[CHOICE 2 ISSUE] MISSING STREAMS')
            else:
                print('ALREADY PROCESSED')
            a = pd.read_csv('stream1.csv', header=None)
            b = pd.read_csv('stream2.csv', header=None)
            c = pd.read_csv('stream3.csv', header=None)
            all_streams.append(pd.Series(a[0]))
            all_streams.append(pd.Series(b[0]))
            all_streams.append(pd.Series(c[0]))

            features_dataframe, combined_features = sheet_generator(all_streams[0], all_streams[1], all_streams[2])
            arr = os.listdir('.')
            if 'ngram_list.csv' not in arr:
                corrector(combined_features_list=combined_features)
            else:
                print('\nAVAILABLE CORRECTED NGRAMS')
        elif choice == 3:
            df2 = pd.read_csv('ngram_list.csv', header=None)
            X_train, y_train = creating_bow(corrected_list=list(df2[0]), features_dataframe=features_dataframe,
                                            max_features=max_feat)

            print("TRAINING SET IS NOW READY")
        elif choice == 4:
            sa, sb, sc, test_set = streamers_test(testB_1)
            ngram_test, test_df = sheet_generator_test(sa, sb, sc, test_set)
            X_test, y_test = creating_bow_test(corrected_list_test=ngram_test, dataframe_test=test_df,
                                               max_features=max_feat)

        elif choice == 5:
            try:
                the_machine(X_train, X_test, y_train, y_test)
            except UnboundLocalError as e:
                print("Execute with choice 3 & 4. Retry Err:", e)
        else:
            return


executor()

# TODO -- Train & Test data to be preprocessed together
# TODO -- No Dependency Relation filter except for Determinant
