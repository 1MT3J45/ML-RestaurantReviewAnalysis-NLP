import progressbar as pb
import numpy as np
import pandas as pd
import ast
import nltk


def unique(whole_df, bigram_col, dep_rel_col):
    '''This function takes Dataframe columns of bigram and dependency relations
    into list and then both of these lists are consolidated into distinct values
    or set'''
    dataframe = whole_df
    u_list = list()
    u_set = ''

    for i in range(len(dataframe)):

        if dataframe.iloc[i, bigram_col] is np.nan and dataframe.iloc[i, dep_rel_col] is np.nan:
            u_set = 'empty'
        elif dataframe.iloc[i, bigram_col] is np.nan:
            u_set = set(dataframe.iloc[i, dep_rel_col].split(';'))
        elif dataframe.iloc[i, dep_rel_col] is np.nan:
            u_set = set(dataframe.iloc[i, bigram_col].split(';'))
        else:
            u_set = set(dataframe.iloc[i, bigram_col].split(';') + dataframe.iloc[i, dep_rel_col].split(';'))

        word_lst = list(u_set)
        u_list.append(word_lst[1:])
        pb.load(i, base=dataframe, text='Unique Features Processing')

    return u_list


def combiner(Feature_df, lemma_col, uniqueFeat_col, use_ast):
    '''Combines the Lemma and Unique features into a single
    sentence. Later to be used for Synonym words using SWordNet
    Dictionary. The Flag use_ast Parameter will decide whether to use
    ast's literal evaluation or not. Literal eval helps to extract
    List from a string data'''
    dataframe = Feature_df
    c_list = list()
    sentence = ''
    for i in range(len(dataframe)):
        if use_ast is True:
            sentence = ast.literal_eval(dataframe.iloc[i, lemma_col]) + list(dataframe.iloc[i, uniqueFeat_col])
        elif use_ast is False:
            sentence = dataframe.iloc[i, lemma_col] + dataframe.iloc[i, uniqueFeat_col]
        # fetching list from string
        if '' in sentence:
            popit = sentence.index('')
            sentence.pop(popit)
        sentence = ';'.join(sentence)
        c_list.append(sentence)
        pb.load(i, dataframe, 'Combining Lemma & Ufeat')
    return c_list


def synset_ngram(ngram):
    syn02 = syn10 = ''
    for i in range(len(ngram)):
        incoming = ngram[i].split(sep=' ')
        print("Iteration ", i)
        if len(incoming) > 1:
            # print('Bigram', incoming[0], incoming[1])
            syn00 = get_correct_syns(word=incoming[0])
            syn01 = get_correct_syns(word=incoming[1])
            syn02 = syn00[0]+' '+syn01[0]
            # print(syn02)
        elif len(incoming) == 1:
            # print('Unigram', incoming[0])
            syn10 = get_correct_syns(incoming[0])
    op = [syn02, syn10]
    return op


def get_correct_syns(word):
    from nltk.corpus import wordnet
    from autocorrect import spell

    synonyms = []
    correct_word = spell(word)
    for syn in wordnet.synsets(correct_word):
        for l in syn.lemmas():
            synonyms.append(l.name())

    if len(synonyms) is 0:
        synonyms.append('blank')

    brk_sent = list(set(synonyms))
    brk_sent = nltk.pos_tag(brk_sent)
    #print(brk_sent)
    synonyms = []
    for w, tag in brk_sent:
        #print(w, " - ", tag)
        if tag.startswith("NN") or tag.startswith("JJ"):
            synonyms.append(w)
        else:
            pass# print(w, "is not an NOUN")
    #print(synonyms)

    return synonyms


def get_correct_spell(word_list, split_by=' '):
    from autocorrect import spell
    li = word_list
    spell_right = list()
    spell_sent = list()
    for i in range(len(li)):
        w_li = word_list[i].split(split_by)
        sent = ''
        spell_right = list()
        # print(w_li)
        for word in w_li:
            if word.startswith("n't"):
                word = 'not' + word[3:]
                cw = spell(word)
            elif word.startswith("I've") or word.startswith("i've"):
                word = 'i have' + word[4:]
                cw = spell(word)
            elif word.startswith("It's") or word.startswith("it's"):
                word = 'it is' + word[4:]
                cw = spell(word)
            else:
                cw = spell(word)
            spell_right.append(cw.lower())
        # print(spell_right)
        sent = ' '.join(spell_right)
        spell_sent.append(sent)
        pb.load(i, li, 'Correcting Words')
    return spell_sent


def get_correct_spell_lemmas(word_list, split_by=' '):
    from autocorrect import spell
    li = word_list
    spell_right = list()
    spell_sent = list()
    for i in range(len(li)):
        w_li = ast.literal_eval(word_list[i])
        sent = ''
        spell_right = list()
        # print(w_li)
        for word in w_li:
            if word.startswith("n't"):
                word = 'not' + word[3:]
                cw = spell(word)
            elif word.startswith("I've") or word.startswith("i've"):
                word = 'i have' + word[4:]
                cw = spell(word)
            elif word.startswith("It's") or word.startswith("it's"):
                word = 'it is' + word[4:]
                cw = spell(word)
            else:
                cw = spell(word)
            spell_right.append(cw.lower())
        # print(spell_right)
        sent = ' '.join(spell_right)
        spell_sent.append(sent)
        pb.load(i, li, 'Correcting Words')
    return spell_sent


def bow_creater(array):
    # Creating Bag of Words Model
    from sklearn.feature_extraction.text import CountVectorizer

    ngram_list = array

    cv = CountVectorizer(max_features=33433, ngram_range=(1, 2))
    X = cv.fit_transform(ngram_list).toarray()
    itemDict = cv.vocabulary_
    print(cv.vocabulary_)
    return X
