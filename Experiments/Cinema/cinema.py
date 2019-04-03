#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################
# Experiment for Figures 13 and 14 in https://arxiv.org/pdf/1901.08949.pdf #
############################################################################

import numpy as np
import pandas as pd
import cupy as cp
from collections import Counter
from heapq import nlargest

import sys
sys.path.insert(0, "../")

from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe
sys.path.insert(0, "Cinema/")


##################################################
# Load the Word2Vec vectors using fasttext
# Please download the file from:
# https://fasttext.cc/docs/en/english-vectors.html
##################################################
import io
def load_vectors(fname, size=None):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 0
    for line in fin:
        if size and i >= size:
            break
        if i >= 2000:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(tokens[1:], dtype='f8')
        i += 1
    return data


dictionnary = load_vectors('Data/dict.vec', size=20000)
dictionnary_pd = pd.DataFrame(dictionnary).T


##################################
# Text Preprocessing
# And transformation into measures
##################################
import string
def textToMeasure(text):
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    words = text.split(' ')
    table = str.maketrans('', '', string.punctuation.replace("'", ""))
    words = [w.translate(table) for w in words if len(w) > 0]
    words = [w for w in words if w in dictionnary.keys()]
    words = [w for w in words if not w[0].isupper()]
    words = [w for w in words if not w.isdigit()]
    cX = Counter(words)
    size = len(words)
    cX = Counter(words)
    words = list(set(words))
    a = np.array([cX[w] for w in words])/size
    X = np.array([dictionnary[w] for w in words])
    return X, a, words


def load_text(file):
    """return X,a,words"""
    with open(file) as fp:
        text = fp.read()
    return textToMeasure(text)


############
# Plotting #
############
def plot_pushforwards_wordcloud(SRW, words_X, words_Y):
    """Plot the projected measures as word clouds."""
    proj_X, proj_Y = SRW.get_projected_pushforwards()
    N_print_words = 30
    plt.figure(figsize=(10,10))
    plt.scatter(proj_X[:,0], proj_X[:,1], s=X.shape[0]*20*a, c='r', zorder=10, alpha=0.)
    plt.scatter(proj_Y[:,0], proj_Y[:,1], s=Y.shape[0]*20*b, c='b', zorder=10, alpha=0.)
    large_a = nlargest(N_print_words, [a[words_X.index(i)] for i in words_X if i not in words_Y])[-1]
    large_b = nlargest(N_print_words, [b[words_Y.index(i)] for i in words_Y if i not in words_X])[-1]
    large_ab = nlargest(N_print_words, [0.5*a[words_X.index(i)] + 0.5*b[words_Y.index(i)] for i in words_Y if i in words_X])[-1]
    for i in range(a.shape[0]):
        if a[i] > large_a:
            if words_X[i] not in words_Y:
                plt.gca().annotate(words_X[i], proj_X[i,:], size=2500*a[i], color='b', ha='center', alpha=0.8)
    for j in range(b.shape[0]):
        if b[j] > large_b and words_Y[j] not in words_X:
            plt.gca().annotate(words_Y[j], proj_Y[j,:], size=2500*b[j], color='r', ha='center', alpha=0.8)
        elif words_Y[j] in words_X and 0.5*b[j]+0.5*a[words_X.index(words_Y[j])] > large_ab:
            size = 0.5*b[j] + 0.5*a[words_X.index(words_Y[j])]
            plt.gca().annotate(words_Y[j], proj_Y[j,:], size=2500*size, color='darkviolet', ha='center', alpha=0.8)
    plt.axis('equal')
    plt.axis('off')
    plt.show()



#########################################################################
# COMPUTE SRW DISTANCES BETWEEN THE MOVIES
# AND PLOT THE OPTIMAL SUBSPACE BETWEEN KILL BILL VOL.1 AND INTERSTELLAR
#########################################################################

scripts = ['DUNKIRK.txt', 'GRAVITY.txt', 'INTERSTELLAR.txt', 'KILL_BILL_VOLUME_1.txt', 'KILL_BILL_VOLUME_2.txt', 'THE_MARTIAN.txt', 'TITANIC.txt']
Nb_scripts = len(scripts)
SRW_matrix = cp.zeros((Nb_scripts, Nb_scripts))
measures = []
for film in scripts:
    measures.append(load_text('Data/'+film))

for film1 in scripts:
    for film2 in scripts:
        i = scripts.index(film1)
        j = scripts.index(film2)
        if i < j:
            X,a,words_X = measures[i]
            Y,b,words_Y = measures[j]
            algo = FrankWolfe(reg=0.1, step_size_0=None, max_iter=50, threshold=0.01, max_iter_sinkhorn=30, threshold_sinkhorn=1e-3, use_gpu=True)
            SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k=2)
            SRW.run()
            SRW.plot_convergence()
            SRW_matrix[i,j] = SRW.get_value()
            SRW_matrix[j,i] = SRW_matrix[i,j]
            print('SRW (', film1, ',', film2, ') =', SRW_matrix[i,j])
            if film2 == 'KILL_BILL_VOLUME_1.txt' and film1 == 'INTERSTELLAR.txt':
                plot_pushforwards_wordcloud(SRW,words_X,words_Y)


# Plot the metric MDS projection of the SRW values
SRW_all = pd.DataFrame(SRW_matrix, index=scripts, columns=scripts)

from sklearn.manifold import MDS
embedding = MDS(n_components=2, dissimilarity='precomputed')
dis = SRW_all-SRW_all[SRW_all>0].min().min()
dis.values[[np.arange(dis.shape[0])]*2] = 0
embedding = embedding.fit(dis)
X = embedding.embedding_

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], alpha=0.)
plt.axis('equal')
plt.axis('off')
c = {'KILL_BILL_VOLUME_1.txt':'red', 'KILL_BILL_VOLUME_2.txt':'red', 'TITANIC.txt':'blue', 'DUNKIRK.txt':'blue', 'GRAVITY.txt':'black', 'INTERSTELLAR.txt':'black', 'THE_MARTIAN.txt':'black'}
for film in scripts:
    i = scripts.index(film)
    plt.gca().annotate(film[:-4].replace('_', ' '), X[i], size=35, ha='center', color=c[film], weight="bold")
plt.show()


# Print the most similar movie to each movie
for film in scripts:
    print('The film most similar to', film[:-4].replace('_', ' '), 'is', SRW_all[film].loc[SRW_all[film]>0].idxmin()[:-4].replace('_', ' '))