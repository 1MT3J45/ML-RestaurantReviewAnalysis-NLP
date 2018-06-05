from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import pandas as pd
import nltk

import re
import spacy

import progressbar as bar

# --------------------- STREAM INITIALIZER ----------------------------
PoS_Tag_sent = list()
S1_corpus = []  # CORPUS (For Collecting Lemmas)
corpora = ''  # CORPORA (For Collecting Corpora of single sentence)
S2_super_corpus = []  # CORPUS (For Collecting Bigrams sentence wise)

# --------------------- SPACY SPECS ------------------------
nlp_en = spacy.load('en_core_web_sm')
plot_nlp = 0  # For Plotting of Dependency chart
S3_dep_corpus = []  # CORPUS (For Collecting Dependency Relations)


# ==================== RULES DEPENDENCY RELATIONS =================
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


# ---------------------------------------------------------- STREAM 1 - LEMMATIZATION
def lemmatize(dataset):
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
        stream1 = pd.Series(S1_corpus)
        return stream1
    except KeyboardInterrupt:
        print("[STAGE 1] Terminating. Human Intervention Not Allowed")
        exit(0)
    except AttributeError as e:
        print("[STAGE 1] Terminating. Due to ", e)
        exit(0)
    # print(S1_corpus)

# ----------------------------------------------------------- STREAM 2 - BIGRAMS

def bigram(dataset):
    try:
        corpora = ''
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
        stream2 = pd.Series(S2_super_corpus)
        return stream2
    except KeyboardInterrupt:
        print("[STAGE 1] Terminating. Human Intervention Not Allowed")
        exit(0)
    except AttributeError as e:
        print("[STAGE 1] Terminating due to",e)
        exit(0)
# ----------------------------------------------------------- STREAM 3 - DEPENDENCY FEATURES (spaCy)


def dep_rel(dataset):
    try:
        corpora = ''
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
        stream3 = pd.Series(S3_dep_corpus)
        return stream3
    except TypeError as e:
        print("[STAGE 2] Unexpected Termination:", e)
        exit(0)
    except KeyboardInterrupt:
        print("[STAGE 2] Human Interrupt Received! Exiting...")
        exit(0)


def syns_of_ngrams(ngram_list):
    syns_book = list()
    syns = list()
    low_synonyms = list()

    for i in range(len(ngram_list)):
        one_review = ngram_list[i]
        for word in one_review:
            syns = list()
            low_synonyms = list()
            for synonyms in wordnet.synsets(word):
                # print(synonyms.lemma_names())
                syns += synonyms.lemma_names()
            syns = list(set(syns))
            # print(syns)
            for j in range(len(syns)):
                # print(syns[j].lower())
                low_synonyms.append(syns[j].lower())
                bar.load(j, base=syns, text='Generating Synonyms')
            ' '.join(low_synonyms)
        syns_book.append(low_synonyms)

    return syns_book

# stream1 = pd.Series(S1_corpus)
# stream2 = pd.Series(S2_super_corpus)
# stream3 = pd.Series(S3_dep_corpus)
# df = pd.concat([stream1, stream2, stream3], axis=1)
# df = df.rename(columns={0: 'lemmas', 1: 'bigrams', 2: 'depenrel'})
# df.to_csv('FeatureSet.csv', index=False)
# df = pd.read_csv('FeatureSet.csv', sep=',')
