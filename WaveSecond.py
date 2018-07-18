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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # 248
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, precision_score, recall_score

#_start = time.time()

testB = pd.read_csv("CSV/Restaurants_Test_Data_phaseB.csv")
trainB = pd.read_csv("CSV/Restaurants_Train_v2.csv")

trainB_1 = trainB.iloc[:, [0, 7, 5]]
testB_1 = testB.iloc[:, [0, 5, 4]]
del testB

fullB = pd.concat([trainB_1, testB_1], axis=0, ignore_index=True)

dataset = fullB  # MAJOR DATA-SET

# --------------------- FUNCTIONS --------------------------


def check_dep_parse(token_dep):
    dep_str = token_dep
    # if dep_str.startswith('nsub'):
    #     pass
    # elif dep_str.startswith('amod'):
    #     pass
    # elif dep_str.startswith('rcmod'):
    #     pass
    # elif dep_str.startswith('dobj'):
    #     pass
    # elif dep_str.startswith('neg'):
    #     pass
    if dep_str.startswith('det'):
        pass
    else:
        return False
    return True


def streamers(full_dataset):
    dataset = full_dataset
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
    stream1 = stream.lemmatize(dataset)

    # ----------------------------------------------------------- STREAM 2 - BIGRAMS
    stream2 = stream.bigram(dataset)

    # ----------------------------------------------------------- STREAM 3 - DEPENDENCY FEATURES (spaCy)
    stream3 = stream.dep_rel(dataset)

    stream1.to_csv('Wave2/stream1.csv', index=False)
    stream2.to_csv('Wave2/stream2.csv', index=False)
    stream3.to_csv('Wave2/stream3.csv', index=False)

    del S1_corpus, S2_super_corpus, S3_dep_corpus

    return stream1, stream2, stream3


def sheet_generator(s1, s2, s3):
    stream1 = s1
    stream2 = s2
    stream3 = s3

    df = pd.concat([stream1, stream2, stream3], axis=1)
    df = df.rename(columns={0: 'lemmas', 1: 'bigrams', 2: 'depenrel'})
    df.to_csv('Wave2/FeatureSet.csv', index=False)
    df = pd.read_csv('Wave2/FeatureSet.csv', sep=',')

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
    whole_df = pd.concat([dataset.iloc[0:, 0], stream1, stream2, stream3, dataset.iloc[0:, 2]], axis=1)
    whole_df = whole_df.rename(columns={'text': 'reviews', 0: 'lemmas', 1: 'bigrams', 2: 'depenrel',
                                        'aspectCategories/aspectCategory/0/_category': 'aspectCategory'})
    whole_df.to_csv('Wave2/WholeSet.csv', index=False)
    whole_df = pd.read_csv('Wave2/WholeSet.csv', sep=',')
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
    Feature_df.to_csv('Wave2/Feature.csv', index=False)
    del whole_df,

    # Aspect Cat, Lemmas + U_feat (from All sentences)
    c_list = list()
    try:
        Feature_df = Feature_df.dropna()
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
        #syn_list = stream.syns_of_ngrams(ngram_list)
        #ngram_list+=syn_list
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

    pd.Series(ngram_list).to_csv('Wave2/ngram_list.csv', index=False)
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
    X = cv.fit_transform(ngram_list).toarray()
    y = Feature_df['aspectCategory']
    del ngram_list
    return X, y, cv.vocabulary_


def split_train_test(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def evaluator(prf, li2, total):
    li = ['Precision', 'Recall\t', 'F1 Measure']
    print("EVALUATION RESULTS".center(60,'_'))
    cmx = [[73.6, 81.3, 90.9, 89.9, 92.2, 87.5],
           [66.1, 70.5, 83.3, 95.2, 89.0, 80.3],
           [69.6, 75.5, 86.9, 92.4, 90.5, 83.5]]
    print('\t\t  %s    %.8s \t %s \t %s \t %s   %s' % (li2[0], li2[1], li2[2], li2[3], li2[4], li2[5]))
    for i in range(len(prf) - 1):
        x = prf[i] * 100.0
        y = cmx[i]
        print('%s \t %r \t\t %r \t %r \t %r \t %r \t   %r' % (li[i], x[0] >= y[0], x[1] >= y[1], x[2] >= y[2],
                                                            x[3] >= y[3], x[4] >= y[4], total[i] >= y[5]))

def prf_to_csv(prf, fileName):
    PRF = np.array(prf)
    PRF_DF = pd.DataFrame(PRF, index=['Precision', 'Recall', 'F1 Measure', 'Support'])
    PRF_DF = PRF_DF.iloc[:,:] * 100
    PRF_DF.to_csv('Results/%s'%fileName)

# ----------------- PREPARING THE MACHINE --------------------------
def the_machine(X_train, X_test, y_train, y_test):
    print("RANDOM FOREST CLASSIFIER RESULTS:")

    rf_Classifier = RandomForestClassifier(n_estimators=50, n_jobs=4)
    rf_Classifier.fit(X_train, y_train)
    y_pred = rf_Classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    prf = precision_recall_fscore_support(y_test, y_pred)
    li2 = list(rf_Classifier.classes_)
    li2.append('TOTAL')

    li = ['Precision', 'Recall\t', 'F1 Measure']
    method = 'weighted'
    total_f1 = f1_score(y_test, y_pred, average=method) * 100
    total_pr = precision_score(y_test, y_pred, average=method) * 100
    total_re = recall_score(y_test, y_pred, average=method) * 100
    total = [total_pr, total_re, total_f1]

    print('\t\t  %s    %.8s \t %s \t %s \t %s   %s' % (li2[0], li2[1], li2[2], li2[3], li2[4], li2[5]))
    for i in range(len(prf) - 1):
        x = prf[i] * 100.0
        print(
            '%s \t %.2f \t\t %.2f \t %.2f \t %.2f \t %.2f \t   %.1f' % (li[i], x[0], x[1], x[2], x[3], x[4], total[i]))
    evaluator(prf, li2, total)
    prf_to_csv(prf, 'RandomForest_LBD.csv')

    print("SVM RESULTS:")
    from sklearn.svm import LinearSVC
    # classifier = SVC(kernel='sigmoid', degree=3)
    linsvc_classifier = LinearSVC(multi_class='crammer_singer', C=1)
    linsvc_classifier.fit(X_train, y_train)
    y_pred = linsvc_classifier.predict(X_test)

    cm1 = confusion_matrix(y_test, y_pred)
    print(cm1)
    prf = precision_recall_fscore_support(y_test, y_pred)
    li2 = list(linsvc_classifier.classes_)
    li2.append('TOTAL')

    li = ['Precision', 'Recall\t', 'F1 Measure']

    total_f1 = f1_score(y_test, y_pred, average=method) * 100
    total_pr = precision_score(y_test, y_pred, average=method) * 100
    total_re = recall_score(y_test, y_pred, average=method) * 100
    total = [total_pr, total_re, total_f1]

    print('\t\t  %s    %.8s \t %s \t %s \t %s   %s' % (li2[0], li2[1], li2[2], li2[3], li2[4], li2[5]))
    for i in range(len(prf) - 1):
        x = prf[i] * 100.0
        print('%s \t %.2f \t\t %.2f \t %.2f \t %.2f \t %.2f \t   %.1f' % (li[i], x[0], x[1], x[2], x[3], x[4], total[i]))
    evaluator(prf, li2, total)
    prf_to_csv(prf, 'LinearSVC_LBD.csv')

    print("MULTINOMIAL NB RESULTS:")
    from sklearn.naive_bayes import MultinomialNB
    # classifier = SVC(kernel='sigmoid', degree=3)
    multi_nb_classifier = MultinomialNB()
    multi_nb_classifier.fit(X_train, y_train)
    y_pred = multi_nb_classifier.predict(X_test)

    cm1 = confusion_matrix(y_test, y_pred)
    print(cm1)
    prf = precision_recall_fscore_support(y_test, y_pred)
    li2 = list(multi_nb_classifier.classes_)
    li2.append('TOTAL')

    li = ['Precision', 'Recall\t', 'F1 Measure']

    total_f1 = f1_score(y_test, y_pred, average=method) * 100
    total_pr = precision_score(y_test, y_pred, average=method) * 100
    total_re = recall_score(y_test, y_pred, average=method) * 100
    total = [total_pr, total_re, total_f1]

    print('\t\t  %s    %.8s \t %s \t %s \t %s   %s' % (li2[0], li2[1], li2[2], li2[3], li2[4], li2[5]))
    for i in range(len(prf) - 1):
        x = prf[i] * 100.0
        print(
            '%s \t %.2f \t\t %.2f \t %.2f \t %.2f \t %.2f \t   %.1f' % (li[i], x[0], x[1], x[2], x[3], x[4], total[i]))
    evaluator(prf, li2, total)
    prf_to_csv(prf, 'MultinomialNB_LBD.csv')

    print("VOTING CLASSIFIER RESULTS:")
    # BEST CLASSIFIERS
    RFC_C1 = RandomForestClassifier(n_estimators=25, n_jobs=4)
    LSVC_C2 = LinearSVC(multi_class='crammer_singer', C=1)
    MNB_C3 = MultinomialNB()
    from sklearn.ensemble import VotingClassifier
    # classifier = GaussianNB()
    # classifier = MultinomialNB(fit_prior=False)
    classifier = VotingClassifier(estimators=[('lr', RFC_C1), ('rf', LSVC_C2),
                                              ('gnb', MNB_C3)], voting='hard', n_jobs=4)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm1 = confusion_matrix(y_test, y_pred)
    print(cm1)
    prf = precision_recall_fscore_support(y_test, y_pred)
    li2 = list(classifier.classes_)
    li2.append('TOTAL')

    li = ['Precision', 'Recall\t', 'F1 Measure']

    total_f1 = f1_score(y_test, y_pred, average='macro') * 100
    total_pr = precision_score(y_test, y_pred, average='micro') * 100
    total_re = recall_score(y_test, y_pred, average='micro') * 100
    total = [total_pr, total_re, total_f1]

    print('\t\t  %s    %.8s \t %s \t %s \t %s   %s' % (li2[0], li2[1], li2[2], li2[3], li2[4], li2[5]))
    for i in range(len(prf) - 1):
        x = prf[i] * 100.0
        print('%s \t %.2f \t\t %.2f \t %.2f \t %.2f \t %.2f \t   %.1f' % (li[i], x[0], x[1], x[2], x[3], x[4],
                                                                          total[i]))
    evaluator(prf, li2, total)
    prf_to_csv(prf, 'VotingClassifier_LBD.csv')


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
    all_streams = list()
    X, y = 0, 0
    max_feat = 1000
    def take_feat():
        max_feat = int(input('Enter No. of features (MIN:MAX) to use in Machine\n (1000:33433) Input:'))
        return max_feat

    while True:
        global fullB, testB_1

        choice = int(input("""\t\t\t\tMENU\n
-------- Data Pre-processing ---------(1 Hr 20 Mins)
1. Perform Lemmatization, Bi-grams formation \n\t\t& Dependency Relations\n
2. Combine into Unique Features (4Secs)\n
3. Create Bag of Words Model (2Secs)\n
-------- Train Test Split ----------(50 Mins)
4. Perform Pre-processing & Processing on Test Set 
-------- MACHINE LEARNING ------
5. Call Machine
6. Exit
\t Choice:"""))
        if choice == 1:
            arr = os.listdir('Wave2')
            exists = [item.startswith('stream') for item in arr if item.startswith('stream')]
            if 'False' in exists:
                a, b, c = streamers(fullB)
                all_streams.append(a)
                all_streams.append(b)
                all_streams.append(c)
            else:
                print('\t\t\t\t\t\tALREADY PROCESSED: GO TO STEP 2')

        elif choice == 2:
            arr = os.listdir('Wave2')
            exists = [item.startswith('stream') for item in arr if item.startswith('stream')]
            if 'False' in exists:
                print('\t\t\t\t\t\t[CHOICE 2] GENERATING STREAMS')
                streamers(fullB)
            else:
                print('\t\t\t\t\t\tALREADY PROCESSED: GO TO STEP 3')
            a = pd.read_csv('Wave2/stream1.csv', header=None)
            b = pd.read_csv('Wave2/stream2.csv', header=None)
            c = pd.read_csv('Wave2/stream3.csv', header=None)
            all_streams.append(pd.Series(a[0]))
            all_streams.append(pd.Series(b[0]))
            all_streams.append(pd.Series(c[0]))

            features_dataframe, combined_features = sheet_generator(all_streams[0], all_streams[1], all_streams[2])
            arr = os.listdir('Wave2')
            if 'ngram_list.csv' not in arr:
                corrector(combined_features_list=combined_features)
            else:
                print('\n\t\t\t\t\t\tAVAILABLE CORRECTED NGRAMS: OFFLINE AVAILABLE')
        elif choice == 3:
            max_feat = take_feat()
            df2 = pd.read_csv('Wave2/ngram_list.csv', header=None)
            X, y, vocab = creating_bow(corrected_list=list(df2[0]), features_dataframe=features_dataframe,
                                            max_features=max_feat)
            print("\t\t\t\t\t\tDATASET IS NOW READY")
            # X_df = pd.DataFrame(X)
            # y_df = pd.DataFrame(y)
            # import operator
            # dataset_cv = pd.concat([X_df, y_df], axis=1)
            # dataset_cv.to_csv('processed_dataset.csv', index=False)
            # pcd_df = pd.read_csv('processed_dataset.csv')
            # sorted_val_df.iloc[:, 0]
            # list(sorted_val_df.iloc[:, 0])
            # pcd_df.columns = list(sorted_val_df.iloc[:, 0]) + ['aspectCategory']
        elif choice == 4:
            # arr = os.listdir('./Wave2/')
            # if 'X_train.csv' and 'X_test.csv' and 'y_train.csv' and 'y_test.csv' in arr:
            #     X_train = np.array(pd.read_csv('Wave2/X_train.csv', header=None))
            #     X_test = np.array(pd.read_csv('Wave2/X_test.csv', header=None))
            #     y_train = np.array(pd.read_csv('Wave2/y_train.csv', header=None))
            #     y_test = np.array(pd.read_csv('Wave2/y_test.csv', header=None))
            #
            #     print('\t'*6,"SPLIT AVAILABLE & COMPLETE")
            # else:
            X_train, X_test, y_train, y_test = split_train_test(X, y)
            print("\t"*6, "TRAIN TEST SPLIT READY")
            # pd.DataFrame(X_train).to_csv('Wave2/X_train.csv', index=False)
            # pd.DataFrame(X_test).to_csv('Wave2/X_test.csv', index=False)
            # pd.DataFrame(y_train).to_csv('Wave2/y_train.csv', index=False)
            # pd.DataFrame(y_test).to_csv('Wave2/y_test.csv', index=False)
        elif choice == 5:
            try:
                the_machine(X_train, X_test, y_train, y_test)
            except UnboundLocalError as e:
                print("Execute with choice 3 & 4. Retry Err:", e)
        else:
            return


executor()

# TODO GRAPHS
# 1. X: AspectCategories Y: Percentages
# I.    Lemmas (Paper vs. Results)
# II.   Lemmas + Dependency
# III.  Lemmas + Bigrams + Dependency

# from collections import Counter
# dictionary = Counter(dataset['aspectCategories/aspectCategory/0/_category'])
# dictionary_ = dict(dictionary)

# f1 = open('category_stack.csv', 'w')
# for k, v in dictionary_.items():
#     print(k, v)
#     f1.write(k + ',' + str(v) + '\n')

# f1.close()