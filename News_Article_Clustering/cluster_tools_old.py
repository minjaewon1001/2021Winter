import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from make_data import *
from scluster import SCluster

def data_tsne(embeddings):
    pca = PCA(n_components=5)
    embeddings_PCA = pca.fit_transform(embeddings)
    dfembeddings_PCA = pd.DataFrame(embeddings_PCA)

    #from sklearn.manifold import TSNE
    X = dfembeddings_PCA
    Xtsne = TSNE(n_components=2).fit_transform(X)
    dftsne = pd.DataFrame(Xtsne)
    dftsne.columns = ['x1','x2']
    
    return dftsne

def kmeans_clustering(embeddings, titles, num_clusters, random_state):
    num_clusters = num_clusters
    clustering_model = KMeans(n_clusters=num_clusters,random_state=random_state)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(titles[sentence_id])

    index = 0
    for i, cluster in enumerate(clustered_sentences):
        print("Cluster "+ str(i+1) +" : 개수는 " +str(len(cluster)) +"입니다.")
        print('--------------------------------')
        for j in cluster:
            print(str(index)+" : " +j)
            index+=1
        print("")
        print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
        print()
        print()

def return_kmeans_clustering(embeddings, num_clusters, random_state):
    num_clusters = num_clusters
    clustering_model = KMeans(n_clusters=num_clusters,random_state=random_state)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    return clustering_model, cluster_assignment

def print_centroid_random(embeddings, titles, num_clusters, random_state):
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
        
        #작은 수 3개 cluster 순서 몇 번째에 있는지 기억
        for j in temp_i[0:3]:
            order.append(i.index(j))
        
        for j in random.sample(temp_i[3:],3):
            rand_order.append(i.index(j))
        
        for j in order:
            min_id[num].append(titles.index(clustered_sentences[num][j]))
        
        for j in rand_order:
            rand_id[num].append(titles.index(clustered_sentences[num][j]))

    for i in range(num_clusters):
        print("#####################Cluster " + str(i+1) + "########################")
        print("●●centroid 주변 3개의 corpus는 다음과 같습니다 : ●●")
        print("--------------------------------------------------------------------")
        for j in min_id[i]:
            print(titles[j])
        print("--------------------------------------------------------------------")
        print("■■random corpus 3개는 다음과 같습니다 : ■■")
        for j in rand_id[i]:
            print(titles[j])
        print()
        print()

def print_centroid_old(embeddings, titles, num_clusters, random_state, centroid_num):
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
    for i in range(num_clusters):
        distances.append([])
        min_id.append([])

    #distance list 만들기
    for i in range(len(embeddings)):
        distances[cluster_assignment[i]].append(np.linalg.norm(centroids[cluster_assignment[i]] - embeddings[i]))
    
    for num, i in enumerate(distances) :
        order = []
        temp_i = sorted(i)
        
        #작은 수 3개 cluster 순서 몇 번째에 있는지 기억
        for j in temp_i[0:centroid_num]:
            order.append(i.index(j))
        for j in order:
            min_id[num].append(titles.index(clustered_sentences[num][j]))

    for i in range(num_clusters):
        print("#####################Cluster " + str(i+1) + "########################")
        print("●●centroid 주변 " + str(centroid_num) + "개의 corpus는 다음과 같습니다 : ●●")
        print("--------------------------------------------------------------------")
        for j in min_id[i]:
            print(titles[j])
        print()
        print()

#sentence clustering 할 때 : cluster size 지나치게 작을 경우 print 하지 않는다.
def print_centroid(embeddings, texts, num_clusters=False, random_state=42, centroid_num=1, cluster_size_option=True):
    if num_clusters == False :
        num_clusters = SCluster(org=3, lim=5).fit(pd.DataFrame(embeddings)).result['optimal_cluster']
    else : 
        num_clusters = num_clusters
    clustering_model = KMeans(n_clusters=num_clusters,random_state=random_state)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(texts[sentence_id])

    centroids = clustering_model.cluster_centers_

    distances = []
    min_id = []
    for i in range(num_clusters):
        distances.append([])
        min_id.append([])

    #distance list 만들기
    for i in range(len(embeddings)):
        distances[cluster_assignment[i]].append(np.linalg.norm(centroids[cluster_assignment[i]] - embeddings[i]))
    
    for num, i in enumerate(distances) :
        order = []
        temp_i = sorted(i)
        
        #작은 수 3개 cluster 순서 몇 번째에 있는지 기억
        for j in temp_i[0:centroid_num]:
            order.append(i.index(j))
        for j in order:
            min_id[num].append(texts.index(clustered_sentences[num][j]))

    #cluster_size_option을 True로 할 경우 cluster size가 작은 경우 출력하지 않게끔 함.
    if cluster_size_option == True :
        idx_list_big_cluster = []
        for idx, i in enumerate(clustered_sentences) :
            #최솟값(절대 수치) 지정하면 좋음. (ex : 5개보다는 크게끔)
            if len(i) >= round(len(texts) * 0.1) :
                idx_list_big_cluster.append(idx)
        min_id = filter_idx_list(min_id, idx_list_big_cluster)
        clustered_sentences = filter_idx_list(clustered_sentences, idx_list_big_cluster)

    #cluster size가 큰 순서대로 출력
    cluster_size_list = []
    for i in clustered_sentences :
        cluster_size_list.append(len(i))
    cluster_size_sort_idx = []
    for i in sorted(cluster_size_list, reverse = True) :
        cluster_size_sort_idx.append(cluster_size_list.index(i))

    for idx, i in enumerate(cluster_size_sort_idx) :
        print("#####################Cluster " + str(idx+1) + "(" +str(len(clustered_sentences[i])) + "개)########################")
        print("●●centroid 주변 " + str(centroid_num) + "개의 corpus는 다음과 같습니다 : ●●")
        print("--------------------------------------------------------------------")
        for j in min_id[i]:
            print(texts[j])
        print()
        print()

def clustering_visualization(embeddings, num_clusters, random_state):
    emb_tsne = data_tsne(embeddings)
    num_clusters = num_clusters
    clustering_model = KMeans(n_clusters=num_clusters, random_state=random_state)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    emb_tsne['labels'] = cluster_assignment
    emb_tsne.columns = ['x1','x2','labels']

    plt.figure(figsize=(12,12))
    
    p_list = ['green', 'red', 'dodgerblue', 'orange', 'purple', 'black']

    sns.scatterplot(x='x1',y='x2', hue = 'labels', data=emb_tsne,legend='full', palette = p_list[:num_clusters], alpha=1.0, s=100)
    

    plt.show()