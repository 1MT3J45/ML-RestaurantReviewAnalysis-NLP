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


def combiner(Feature_df, lemma_col, uniqueFeat_col):
    '''Combines the Lemma and Unique features into a single
    sentence. Later to be used for Synonym words using SWordNet
    Dictionary'''
    dataframe = Feature_df
    c_list = list()
    for i in range(len(dataframe)):
        sentence = ast.literal_eval(dataframe.iloc[i,lemma_col]) + dataframe.iloc[i, uniqueFeat_col]
        sentence = ' '.join(sentence)
        c_list.append(sentence)
        pb.load(i,dataframe,'Combining Lemma & Ufeat')
    return c_list
