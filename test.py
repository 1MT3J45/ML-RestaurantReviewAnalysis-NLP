from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from multiprocessing.dummy import Pool as ThreadPool

import os
import pandas as pd
import nltk

import re
import spacy
import progressbar as bar

testB = pd.read_csv("CSV/Restaurants_Test_Data_phaseB.csv")
trainB = pd.read_csv("CSV/Restaurants_Train_v2.csv")

# print(testB.head(5))
# print(trainB.head(5))

trainB_1 = trainB.iloc[:, [0, 7, 5]]
testB_1 = testB.iloc[:, [0, 5, 4]]

fullB = pd.concat([testB_1, trainB_1], axis=0, ignore_index=True)

nltk.download('stopwords')


dataset = fullB     # MAJOR DATA-SET
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
S1_corpus = []          # CORPUS (For Collecting Lemmas)
corpora = ''            # CORPORA (For Collecting Corpora of single sentence)
S2_super_corpus =[]     # CORPUS (For Collecting Bigrams sentence wise)

# --------------------- SPACY SPECS ------------------------
nlp_en = spacy.load('en_core_web_sm')
plot_nlp = 0            # For Plotting of Dependency chart
S3_dep_corpus = []      # CORPUS (For Collecting Dependency Relations)

# ---------------------------------------------------------- STREAM 1 - LEMMATIZATION
try:
    for i in range(0, len(dataset)):
        review = re.sub("[^a-zA-Z]", ' ', dataset['text'][i])
        review = review.lower()
        review = review.split()

        # Learn More at https://www.quora.com/What-is-difference-between-stemming-and-lemmatization
        ps = PorterStemmer()
        nl = WordNetLemmatizer()

        review = [ps.stem(nl.lemmatize(word, pos='v')) for word in review if not word in set(stopwords.words('english'))]
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
            if tag1.startswith('JJ') and tag2.startswith('NN'):     # R1
                corpora += w1+' '+w2+';'
            elif tag1.startswith('RB') and tag2.startswith('JJ'):   # R2
                corpora += w1+' '+w2+';'
            elif tag1.startswith('JJ') and tag2.startswith('JJ'):   # R3
                corpora += w1+' '+w2+';'
            elif tag1.startswith('NN') and tag2.startswith('JJ'):   # R4
                corpora += w1+' '+w2+';'
            elif tag1.startswith('RB') and tag2.startswith('VB'):   # R5
                corpora += w1+' '+w2+';'
            elif tag1.startswith('VB') and tag2.startswith('NN'):   # R6
                corpora += w1+' '+w2+';'
            elif tag1.startswith('JJ') and tag2.startswith('VB'):   # R7
                corpora += w1+' '+w2+';'
            elif tag1.startswith('RB') and tag2.startswith('RB'):   # R8
                corpora += w1+' '+w2+';'
            elif tag1.startswith('RB') and tag2.startswith('VB'):   # R9
                corpora += w1+' '+w2+';'

        S2_super_corpus.append(corpora)
        corpora = ''
        bar.load(i, base=dataset, text='Stream 2')
    print('Stream 2: Processed')
except KeyboardInterrupt:
        print("Terminating. Human Intervention Not Allowed")
        exit(0)
# ----------------------------------------------------------- STREAM 3 - DEPENDENCY FEATURES (spaCy)

try:
    for increment in range(len(dataset)):
        sentence = dataset.iloc[increment, 0].lower()
        #print(increment)
        for token in nlp_en(sentence):
            dep = check_dep_parse(token.dep_)
            if dep is True:
                # print(token.dep_, end="> ")
                # print(token.head, token)
                corpora += str(token.head)+' '+str(token)+';'
            else:
                pass
        S3_dep_corpus.append(corpora)
        corpora = ''
        bar.load(increment,base=dataset, text='Stream 3')
    print('Stream 3: Processed')
    plot_nlp = nlp_en(sentence)

    pass
except TypeError as e:
    print("Unexpected Termination:", e)
    exit(0)
except KeyboardInterrupt:
    print("Human Interrupt Received! Exiting...")
    exit(0)

stream1 = pd.Series(S1_corpus)
stream2 = pd.Series(S2_super_corpus)
stream3 = pd.Series(S3_dep_corpus)
df = pd.concat([stream1, stream2, stream3], axis=1)
df = df.rename(columns={0: 'lemmas', 1: 'bigrams', 2: 'depenrel'})
df.to_csv('FeatureSet.csv', index=False)
df =pd.read_csv('FeatureSet.csv', sep=',')
try:
    pool = ThreadPool(2)
    pool.map(os.system('firefox localhost:5000 &'), spacy.displacy.serve(plot_nlp, style='dep')).join()
    exit(0)
except OSError:
    print("Browser must start with Graph. If doesn't please make sure to use Ubuntu with Firefox")
except TypeError:
    print("Browser must start with Graph. If doesn't please make sure to use Ubuntu with Firefox")


# # Set up spaCy
# from spacy.lang.en import English
# parser = English()
#
# # Test Data
# multiSentence = "There is an art, it says, or rather, a knack to flying." \
#                  "The knack lies in learning how to throw yourself at the ground and miss." \
#                  "In the beginning the Universe was created. This has made a lot of people "\
#                  "very angry and been widely regarded as a bad move."
#
# # all you have to do to parse text is this:
# #note: the first time you run spaCy in a file it takes a little while to load up its modules
# parsedData = parser(multiSentence)
#
# # Let's look at the tokens
# # All you have to do is iterate through the parsedData
# # Each token is an object with lots of different properties
# # A property with an underscore at the end returns the string representation
# # while a property without the underscore returns an index (int) into spaCy's vocabulary
# # The probability estimate is based on counts from a 3 billion word
# # corpus, smoothed using the Simple Good-Turing method.
# for i, token in enumerate(parsedData):
#     print("original:", token.orth, token.orth_)
#     print("lowercased:", token.lower, token.lower_)
#     print("lemma:", token.lemma, token.lemma_)
#     print("shape:", token.shape, token.shape_)
#     print("prefix:", token.prefix, token.prefix_)
#     print("suffix:", token.suffix, token.suffix_)
#     print("log probability:", token.prob)
#     print("Brown cluster id:", token.cluster)
#     print("----------------------------------------")
#     if i > 1:
#         break
#
# # Let's look at the sentences
# sents = []
# parsedData.is_parsed = True
# # the "sents" property returns spans
# # spans have indices into the original string
# # where each index value represents a token
# for span in parsedData.sents:
#     # go from the start to the end of each span, returning each token in the sentence
#     # combine each token using join()
#     sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
#     sents.append(sent)
#
# for sentence in sents:
#     print(sentence)
#
# # Let's look at the part of speech tags of the first sentence
# for span in parsedData.sents:
#     sent = [parsedData[i] for i in range(span.start, span.end)]
#     break
#
# for token in sent:
#     print(token.orth_, token.pos_)
# # ----------------------------------------------------
# import spacy
# nlp = spacy.load("en")
#
# document = "There is an art, it says, or rather, a knack to flying." \
#          "The knack lies in learning how to throw yourself at the ground and miss." \
#          "In the beginning the Universe was created. This has made a lot of people "\
#          "very angry and been widely regarded as a bad move."
#
# document = nlp(document)
# dir(document)
#
# # Getting sentences:
# list(document.sents)
#
# # Part of Speech tagging
# all_tags = {w.pos: w.pos_ for w in document}
#
# # All tags of first sentence of our document
# for word in list(document.sents)[0]:
#     print(word, word.tag_)
#
# # Defining some parameters
# noisy_pos_tags = ["PROP"]
# min_token_length = 2
#
# # Dependency Parsing
# # Extracting all review sentences that contains the term 'art'
# art = [sent for sent in document.sents if 'art' in sent.string.lower()]
#
# # create dependency tree
# sentence = art[0]
# for word in sentence:
#     print(word, ': ', str(list(word.children)))
