import re
import numpy as np
#from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import os
from nltk.tokenize import sent_tokenize
from scipy import spatial
import copy
import json
from scluster import SCluster
import pandas as pd

#중복제거 하는 모듈 title 같거나 content 같으면 삭제 (title 달라도 content 일치하면 삭제)
def delete_duplication(title_list, content_list, passage_id_list) :
    title_list_no_dup_1 = [] #첫 번째 title_list : title 기준으로만 filtering
    index_list = []
    n = 0
    for i in title_list:
        index_list.append(n)
        if i not in title_list_no_dup_1:
            title_list_no_dup_1.append(i)  
        else :
            index_list = index_list[:-1]
        n += 1
    content_list_no_dup = [] #content 중심으로 filtering
    title_list_no_dup_2 = []
    passage_id_list_no_dup = []
    for i in index_list :
        if content_list[i] not in content_list_no_dup :
            content_list_no_dup.append(content_list[i])
            title_list_no_dup_2.append(title_list[i])
            passage_id_list_no_dup.append(passage_id_list[i])
    return title_list_no_dup_2, content_list_no_dup, passage_id_list_no_dup

#text 전처리 모듈 (현재 버전) (특수기호 제거 부분 제외했음.)
def clean_text(text):
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' 
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("E-mail제거 : " , text , "\n")
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("URL 제거 : ", text , "\n")
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("한글 자음 모음 제거 : ", text , "\n")
    pattern = '<[^>]*>'        
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("태그 제거 : " , text , "\n")
    pattern = r'\([^)]*\)'
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("괄호와 괄호안 글자 제거 :  " , text , "\n")
    pattern = '[^\w\s]'
    #pattern = r'[●■ⓒ-=+#/\:^@*\"※~ㆍ』‘|\(\)\[\]`\'…》\”\“\’·]'
    text = re.sub(pattern=pattern, repl='', string=text)
    #print("특수기호 제거 : ", text , "\n" )
    text = text.strip('\n')
    text = text.strip()
    #print("양 끝 공백 제거 : ", text , "\n" )
    text = " ".join(text.split())
    #print("중간에 공백은 1개만 : ", text )
    return text

def sbert_embedding_text_list(text_list, embedder) :
    embedding_list = []
    for i in text_list :
        if i == "" :
            embedding_list.append([0]*768)
        else :
            embedding_list.append(embedder.encode(i).tolist())
    return embedding_list

def filter_idx_list(text_list, idx_list) :
    new_text_list = []
    for i in idx_list :
        new_text_list.append(text_list[i])
    return new_text_list

def titles_clustering(embedding_list, passage_id_list, clustering_method, random_state_option, number_of_clusters_option, centroid_number_option, cluster_size_option) :
    if clustering_method == "KMeans" :
        if number_of_clusters_option == False :
            num_clusters = SCluster(org=3, lim=5).fit(pd.DataFrame(embedding_list)).result['optimal_cluster']
        else :
            num_clusters = number_of_clusters_option
        clustering_model = KMeans(n_clusters=num_clusters, random_state=random_state_option)
        clustering_model.fit(embedding_list)
        cluster_assignment = clustering_model.labels_.tolist()
        clustered_id_list = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_id_list[cluster_id].append(passage_id_list[sentence_id])
        clustered_id_list_sorted = sorted(clustered_id_list, reverse=True,key=len)
        size_index = []
        for idx, i in enumerate(clustered_id_list_sorted) :
            size_index.append(clustered_id_list_sorted.index(clustered_id_list[idx]))
        cluster_assignment_copied = copy.deepcopy(cluster_assignment)
        for idx, i in enumerate(cluster_assignment_copied) :
            cluster_assignment[idx] = size_index[i]
        
        centroids = clustering_model.cluster_centers_
        centroids_sorted = []
        for i in size_index :
            centroids_sorted.append(centroids[i])
        distances = []
        min_id = []
        for i in range(num_clusters):
            distances.append([])
            min_id.append([])
        for i in range(len(embedding_list)):
            distances[cluster_assignment[i]].append(np.linalg.norm(centroids_sorted[cluster_assignment[i]] - embedding_list[i]))
        for num, i in enumerate(distances) :
            order = []
            temp_i = sorted(i)
            
            #작은 수 n개 cluster 순서 몇 번째에 있는지 기억
            for j in temp_i[0:centroid_number_option]:
                order.append(i.index(j))
            for j in order:
                min_id[num].append(passage_id_list.index(clustered_id_list_sorted[num][j]))

        str_cluster_downsized = "cluster is not downsized"
        #cluster_size_option을 True로 할 경우 cluster size가 작은 경우 or 5개보다 작은 경우 출력하지 않게끔 함.
        if cluster_size_option == True :
            for idx, i in enumerate(clustered_id_list_sorted) :
                if len(i) <= round(len(embedding_list) * 0.1) or len(i) <= 5:
                    clustered_id_list_sorted = clustered_id_list_sorted[:idx]
                    centroids_sorted = centroids_sorted[:idx]
                    min_id = min_id[:idx]
            if num_clusters > len(min_id) :
                str_cluster_downsized = "cluster is downsized from " + str(num_clusters) + " to " + str(len(min_id)) + "."
            
        
        return cluster_assignment, clustered_id_list_sorted, centroids_sorted, min_id, str_cluster_downsized

def first_sentence_text_list(text_list) : 
    index = 0
    while(True) :
        if len(text_list[index]) >= 10 :
            break
        else : 
            if index == len(text_list) - 1 :
                index = 0
                for i in text_list :
                    if len(i) != 0 :
                        break
                    else :
                        index += 1
            index += 1
    return text_list[index], index

def longest_sentence_text_list(text_list) : 
    len_text_list = []
    for i in text_list :
        len_text_list.append(len(i))
    index = len_text_list.index(max(len_text_list))
    return text_list[index], index

def most_similar_with_title_embedding_list(embedding_list, title_embedding) :
    similarity_list = []
    for i in embedding_list :
        similarity_list.append(np.linalg.norm(np.array(i) - np.array(title_embedding)))
    index = similarity_list.index(max(similarity_list))
    return index

def sentences_clustering(candidate_dict_list, centroid_title_embedding, clustering_method, \
                            random_state_option, number_of_clusters_option, centroid_number_option, cluster_size_option) :
    new_dict_list = []
    for i in candidate_dict_list : 
        if len(i['text_processed']) >= 15 :
            new_dict_list.append(i)
    
    new_dict_list_f = []
    for i in new_dict_list :
        if spatial.distance.cosine(centroid_title_embedding, i['text_embedding']) >= 0.2 :
            new_dict_list_f.append(i)

    embedding_list = []
    passage_id_from_list = []
    for i in new_dict_list_f :
        embedding_list.append(i['text_embedding'])
        passage_id_from_list.append(i['passage_id_from'])

    if clustering_method == "KMeans" :
        if number_of_clusters_option == False :
            num_clusters = SCluster(org=2, lim=3).fit(pd.DataFrame(embedding_list)).result['optimal_cluster']
        else :
            num_clusters = number_of_clusters_option
        clustering_model = KMeans(n_clusters=num_clusters, random_state=random_state_option)
        clustering_model.fit(embedding_list)
        cluster_assignment = clustering_model.labels_.tolist()
        clustered_id_list = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_id_list[cluster_id].append(passage_id_from_list[sentence_id])
        clustered_id_list_sorted = sorted(clustered_id_list, reverse=True,key=len)
        size_index = []
        for idx, i in enumerate(clustered_id_list_sorted) :
            size_index.append(clustered_id_list_sorted.index(clustered_id_list[idx]))
        cluster_assignment_copied = copy.deepcopy(cluster_assignment)
        for idx, i in enumerate(cluster_assignment_copied) :
            cluster_assignment[idx] = size_index[i]
        
        centroids = clustering_model.cluster_centers_
        centroids_sorted = []
        for i in size_index :
            centroids_sorted.append(centroids[i].tolist())
        distances = []
        min_id = []
        for i in range(num_clusters):
            distances.append([])
            min_id.append([])
        for i in range(len(embedding_list)):
            distances[cluster_assignment[i]].append(np.linalg.norm(np.array(centroids_sorted[cluster_assignment[i]]) - np.array(embedding_list[i])))
            #distances[cluster_assignment[i]].append(np.linalg.norm(centroids_sorted[cluster_assignment[i]] - embedding_list[i]))
        for num, i in enumerate(distances) :
            order = []
            temp_i = sorted(i)
            
            #작은 수 n개 cluster 순서 몇 번째에 있는지 기억
            for j in temp_i[0:centroid_number_option]:
                order.append(i.index(j))
            for j in order:
                min_id[num].append(passage_id_from_list.index(clustered_id_list_sorted[num][j]))

        str_cluster_downsized = "cluster is not downsized"
        #cluster_size_option을 True로 할 경우 cluster size가 작은 경우 or 5개보다 작은 경우 출력하지 않게끔 함.
        if cluster_size_option == True :
            for idx, i in enumerate(clustered_id_list_sorted) :
                if len(i) <= round(len(embedding_list) * 0.1) or len(i) <= 2:
                    clustered_id_list_sorted = clustered_id_list_sorted[:idx]
                    centroids_sorted = centroids_sorted[:idx]
                    min_id = min_id[:idx]
            if num_clusters > len(min_id) :
                str_cluster_downsized = "cluster is downsized from " + str(num_clusters) + " to " + str(len(min_id)) + "."
        return passage_id_from_list, centroids_sorted, min_id, str_cluster_downsized


    
    
    
    
    

    
    
