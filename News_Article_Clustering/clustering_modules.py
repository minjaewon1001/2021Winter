import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import os
import warnings
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from clustering_modules import *
############################################################################

#KoSentenceBERT Embedding
def sbert_titles_embedding(titles, embedder) :
    embeddings = embedder.encode(titles)
    return embeddings

#k-means clustering하고 clustered sentences를 return
def kmeans_clustering(embeddings, titles, num_clusters, random_state):
    num_clusters = num_clusters
    clustering_model = KMeans(n_clusters=num_clusters,random_state=random_state)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(titles[sentence_id])
    index = 0
    cluster_size = []
    for i, cluster in enumerate(clustered_sentences):
        cluster_size.append(len(cluster))
    return cluster_size, clustered_sentences

#kmeans clustering을 수행하고 centroid 3개와 random한 corpus 3개를 return
def centroid_random(embeddings, titles, centroid_number, num_clusters, random_state):
    num_clusters = num_clusters
    clustering_model = KMeans(n_clusters=num_clusters,random_state=random_state)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(titles[sentence_id])
    centroids = clustering_model.cluster_centers_
    distances = []
    min_id = []
    rand_id = []
    for i in range(num_clusters):
        distances.append([])
        min_id.append([])
        rand_id.append([])
    #distance list 만들기
    for i in range(len(embeddings)):
        distances[cluster_assignment[i]].append(np.linalg.norm(centroids[cluster_assignment[i]] - embeddings[i]))
    for num, i in enumerate(distances) :
        order = []
        rand_order = []
        temp_i = sorted(i)
        #작은 수 ~개 cluster 순서 몇 번째에 있는지 기억
        for j in temp_i[0:centroid_number]:
            order.append(i.index(j))
        for j in random.sample(temp_i[centroid_number:],3):
            rand_order.append(i.index(j))
        for j in order:
            min_id[num].append(titles.index(clustered_sentences[num][j]))
        for j in rand_order:
            rand_id[num].append(titles.index(clustered_sentences[num][j]))

    centroid_titles = []
    rand_titles = []
    for i in range(num_clusters) :
        centroid_titles.append([])
        rand_titles.append([])
    for i in range(num_clusters):
        for j in min_id[i]:
            centroid_titles[i].append(titles[j])
        for j in rand_id[i]:
            rand_titles[i].append(titles[j])
    return centroid_titles, rand_titles