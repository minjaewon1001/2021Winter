import re
import numpy as np
from sklearn.cluster import KMeans
import os
from nltk.tokenize import sent_tokenize
from scipy import spatial
import json
import time
from scluster import SCluster
from cluster_tools import *
#############
from flask import Flask, Response, request
from flask import request
from nltk.tokenize import sent_tokenize


def first_module(data) :
    start = time.time()
    title_list = []
    content_list = []
    passage_id_list = []
    for i in data :
        title_list.append(i.get('title'))
        content_list.append(i.get('content'))
        passage_id_list.append(i.get('passage_id'))

    title_list, content_list, passage_id_list = delete_duplication(title_list,content_list,passage_id_list) #중복제거
    title_pro_list = []
    content_pro_list = []
    for i in range(len(title_list)) :
        title_pro_list.append(clean_text(title_list[i]))
        content_pro_list.append(clean_text(content_list[i])) #특수문자, 연속 공백 제거 등의 텍스트 전처리
    output_1st_data_list = []
    for i in range(len(title_list)) :
        output_1st_data_dict = [{
            'title' : {
                        'text' : title_list[i], 
                        'text_processed' : title_pro_list[i],
                        }, 
            'content' : {
                        'text' : content_list[i],
                        'text_processed' : content_pro_list[i]
                        },
            'passage_id' : passage_id_list[i]
        }]
        output_1st_data_list += output_1st_data_dict
    end = time.time()
    output_1st = {'data' : output_1st_data_list,
                    'post_status' : {
                        'error' : None, 
                        'current_documents_number_after_deleting_duplications' : len(title_list),
                        'time' : end - start
            }}
    return output_1st

def second_module(output_1st, option, embedder) :
    start = time.time()
    data = output_1st.get('data')
    title_list = []
    content_list = []
    passage_id_list = []
    title_pro_list = []
    content_pro_list = []
    for i in data :
        title_list.append(i['title'].get('text'))
        title_pro_list.append(i['title'].get('text_processed'))
        content_list.append(i['content'].get('text'))
        content_pro_list.append(i['content'].get('text_processed'))
        passage_id_list.append(i.get('passage_id'))

    if len(title_list) < 30: #중복 제거 후 document수가 30개 이하일 땐 clustering 불가.                            
        end = time.time()
        output_2nd = {'clustering' : None,
                     'post_status' : {
                          'error' : "violate minimum document number(=30) after deleting duplicated documents", 
                          'current_documents' : len(title_list),
                          'time' : output_1st.get('post_status').get('time') + end - start }
        }
        return output_2nd
    else :
        title_embedding_list = sbert_embedding_text_list(title_pro_list, embedder)
        clustering_method = option.get('clustering_method')
        number_of_clusters_option = option.get('assign_num_clusters')
        if option.get('random_state') == False :
            random_state_option = 42
        else :
            random_state_option = option.get('random_state')
        if option.get('assign_centroid_num') == False :
            centroid_number_option = 1
        else : 
            centroid_number_option = option.get('assign_centroid_num')
        cluster_size_option = option.get('cluster_size_option')
        
        cluster_assignment, clustered_id_list, centroid_list, min_id, str_cluster_downsized = \
            titles_clustering(title_embedding_list, passage_id_list, clustering_method, random_state_option, \
                                number_of_clusters_option, centroid_number_option, cluster_size_option)
        clustered_title_list = [[] for i in range(len(clustered_id_list))]
        clustered_title_pro_list = [[] for i in range(len(clustered_id_list))]
        clustered_title_embedding_list = [[] for i in range(len(clustered_id_list))]
        clustered_content_list = [[] for i in range(len(clustered_id_list))]
        clustered_content_pro_list = [[] for i in range(len(clustered_id_list))]
        clustered_passage_id_list = [[] for i in range(len(clustered_id_list))]
        for idx, i in enumerate(cluster_assignment) :
            if i >= len(clustered_id_list) :
                continue
            else :
                clustered_title_list[i].append(title_list[idx])
                clustered_title_pro_list[i].append(title_pro_list[idx])
                clustered_title_embedding_list[i].append(title_embedding_list[idx])
                clustered_content_list[i].append(content_list[idx])
                clustered_content_pro_list[i].append(content_pro_list[idx])
                clustered_passage_id_list[i].append(passage_id_list[idx])

        centroid_title_list = [[] for i in range(len(clustered_id_list))]
        centroid_title_pro_list = [[] for i in range(len(clustered_id_list))]
        centroid_title_embedding_list = [[] for i in range(len(clustered_id_list))]
        centroid_content_list = [[] for i in range(len(clustered_id_list))]
        centroid_content_pro_list = [[] for i in range(len(clustered_id_list))]
        centroid_passage_id_list = [[] for i in range(len(clustered_id_list))]

        for idx, i in enumerate(min_id) :
            for j in i :
                centroid_title_list[idx].append(title_list[j])
                centroid_title_pro_list[idx].append(title_pro_list[j])
                centroid_title_embedding_list[idx].append(title_embedding_list[j])
                centroid_content_list[idx].append(content_list[j])
                centroid_content_pro_list[idx].append(content_pro_list[j])
                centroid_passage_id_list[idx].append(passage_id_list[j])
        clustered_data_number = 0
        for i in clustered_title_list :
            for j in i :
                clustered_data_number += 1
        output_2nd_data_list  = [[] for i in range(len(clustered_id_list))]
        for i in range(len(clustered_id_list)) :
            for j in range(len(clustered_id_list[i])) :
                cluster_data_dict = {
                                    'title' : {
                                                'text' : clustered_title_list[i][j], 
                                                'text_processed' : clustered_title_pro_list[i][j], 
                                                'text_embedding' : clustered_title_embedding_list[i][j]},
                                    'content' : {
                                                'text' : clustered_content_list[i][j],
                                                'text_processed' : clustered_content_pro_list[i][j] },
                                    'passage_id' :clustered_passage_id_list[i][j]}
                output_2nd_data_list[i].append(cluster_data_dict)
        output_2nd_centroid_data_list  = [[] for i in range(len(clustered_id_list))]
        for i in range(len(centroid_title_list)) :
            for j in range(len(centroid_title_list[i])) :
                centroid_data_dict = {
                                    'title' : {
                                                'text' : centroid_title_list[i][j], 
                                                'text_processed' : centroid_title_pro_list[i][j], 
                                                'text_embedding' : centroid_title_embedding_list[i][j]},
                                    'content' : {
                                                'text' : centroid_content_list[i][j], 
                                                'text_processed' : centroid_content_pro_list[i][j]},
                                    'passage_id' :centroid_passage_id_list[i][j]}
                output_2nd_centroid_data_list[i].append(centroid_data_dict)
        output_2nd_clustering_result_list = []
        for i in range(len(clustered_id_list)) :
            output_2nd_clustering_result_dict = {
                'cluster_number' : i+1,
                'cluster_size' :  len(output_2nd_data_list[i]),
                'cluster_data' : output_2nd_data_list[i],
                'centroid_result' : {
                                    'centroid_coordinate' : centroid_list[i].tolist(), 
                                    'centroid_data' : output_2nd_centroid_data_list[i]
                                        }
            }
            output_2nd_clustering_result_list.append(output_2nd_clustering_result_dict)
        end = time.time()
        output_2nd = {'clustering' : 
                        {
                        'clustering_tool' : clustering_method,
                        'random_state_number' : random_state_option,
                        'number_of_clusters' : len(clustered_id_list),
                        'output_centroid_number' : centroid_number_option,
                        'cluster_size_option' : str_cluster_downsized,
                        'clustering_result' : output_2nd_clustering_result_list
                        },
                     'post_status' : {
                          'error' : None, #violate minimum document number(=30) after deleting duplicated documents
                          'current_documents' : clustered_data_number,
                          'time_1st_module' : output_1st.get('post_status').get('time'),
                          'time_2nd_module' : end - start,
                          'time_total' : output_1st.get('post_status').get('time')+ end - start
              }
        }
        return output_2nd

def third_module(output_2nd, option, embedder) :
    start = time.time()
    method = option.get('representative_sentence_selecting_method')
    data = output_2nd.get('clustering').get('clustering_result')
    clustering_method = option.get('clustering_method')
    number_of_clusters_option = option.get('assign_num_clusters')
    if option.get('random_state') == False :
        random_state_option = 42
    else :
        random_state_option = option.get('random_state')
    if option.get('assign_centroid_num') == False :
        centroid_number_option = 1
    else : 
        centroid_number_option = option.get('assign_centroid_num')
    cluster_size_option = option.get('cluster_size_option')


    #키 추가
    for idx1, i in enumerate(data) :
        cluster_centroid_coordinate = i.get('centroid_result').get('centroid_coordinate')
        if method == 'first' or method == 'longest':  
            for idx2, j in enumerate(i['cluster_data']) :
                content_sentence_list = sent_tokenize(j.get('content').get('text'))
                content_sentence_processed_list = []
                for sent in content_sentence_list :
                    content_sentence_processed_list.append(clean_text(sent))
                data[idx1]['cluster_data'][idx2]['split_sentence'] = {
                    'text_list' : content_sentence_list, 
                    'text_processed_list' : content_sentence_processed_list, 
                    'text_embedding_list' : []
                }
                if method == 'first' : 
                    first_sentence, index = first_sentence_text_list(data[idx1]['cluster_data'][idx2]['split_sentence'].get('text_processed_list'))
                    data[idx1]['cluster_data'][idx2]['candidate_sentence'] = {
                    'text' : data[idx1]['cluster_data'][idx2]['split_sentence'].get('text_list')[index], 
                    'text_processed' : first_sentence,
                    'text_index' : index, 
                    'text_embedding' : embedder.encode(first_sentence).tolist(), 
                    'passage_id_from' : j.get('passage_id')
                }
                elif method == 'longest' : 
                    longest_sentence, index = longest_sentence_text_list(data[idx1]['cluster_data'][idx2]['split_sentence'].get('text_list'))
                    data[idx1]['cluster_data'][idx2]['candidate_sentence'] = {
                    'text' : data[idx1]['cluster_data'][idx2]['split_sentence'].get('text_list')[index], 
                    'text_processed' : longest_sentence,
                    'text_index' : index, 
                    'text_embedding' : embedder.encode(longest_sentence).tolist(), 
                    'passage_id_from' : j.get('passage_id')
                    }
                    
        else : #similarity
            for idx2, j in enumerate(i['cluster_data']) :
                content_sentence_list = sent_tokenize(j.get('content').get('text'))
                content_sentence_processed_list = []
                for sent in content_sentence_list :
                    content_sentence_processed_list.append(clean_text(sent))
                content_sentence_embedding_list = []
                for sent in content_sentence_processed_list :
                    if sent == "" :
                        content_sentence_embedding_list.append([0]*768)
                    else :
                        content_sentence_embedding_list.append(embedder.encode(sent).tolist())
                data[idx1]['cluster_data'][idx2]['split_sentence'] = {
                    'text_list' : content_sentence_list, 
                    'text_processed_list' : content_sentence_processed_list, 
                    'text_embedding_list' : content_sentence_embedding_list
                }
                index = most_similar_with_title_embedding_list(content_sentence_embedding_list, j.get('title').get('text_embedding'))
                
                data[idx1]['cluster_data'][idx2]['candidate_sentence'] = {
                    'text' : content_sentence_list[index], 
                    'text_processed' : content_sentence_processed_list[index],
                    'text_index' : index, 
                    'text_embedding' : content_sentence_embedding_list[index], 
                    'passage_id_from' : j.get('passage_id')
                }
        candidate_dict_list = []
        for k in i['cluster_data'] :
            candidate_dict_list.append(k['candidate_sentence'])

        passage_id_from_list, centroids, min_id, str_cluster_downsized = \
            sentences_clustering(candidate_dict_list, cluster_centroid_coordinate, clustering_method, \
                                random_state_option, number_of_clusters_option, centroid_number_option, cluster_size_option)

        clustered_sentence_list = [[] for i in range(len(min_id))]
        clustered_sentence_processed_list = [[] for i in range(len(min_id))]
        clustered_sentence_embedding_list = [[] for i in range(len(min_id))]
        clustered_sentence_index_list = [[] for i in range(len(min_id))]
        for idx, i in enumerate(min_id) :
            for j in i :
                dict_gen = (item for item in candidate_dict_list if item['passage_id_from'] == passage_id_from_list[j])
                dict_find = next(dict_gen, False)
                if dict_find != False :
                    clustered_sentence_list[idx].append(dict_find.get('text'))
                    clustered_sentence_processed_list[idx].append(dict_find.get('text_processed'))
                    clustered_sentence_embedding_list[idx].append(dict_find.get('text_embedding'))
                    clustered_sentence_index_list[idx].append(dict_find.get('text_index'))

        topic_data_dict_list = [[] for i in range(len(min_id))]
        data[idx1]['topic_sentence'] = []
        for j in range(len(min_id)) :
            for k in range(len(min_id[j])) :
                topic_data_dict = {
                    'text' : clustered_sentence_list[j][k],
                    'text_processed' :clustered_sentence_processed_list[j][k],
                    'text_embedding' : clustered_sentence_embedding_list[j][k],
                    'centroid_similarity' : spatial.distance.cosine(clustered_sentence_embedding_list[j][k], cluster_centroid_coordinate),
                    'passage_id_from' : passage_id_from_list[min_id[j][k]]
                }
                topic_data_dict_list[j].append(topic_data_dict)
            topic_sentence_cluster_dict = {
                'cluster_number' : j+1,
                'centroid_coordinate' : centroids[j],
                'data' : topic_data_dict_list[j]
            }
            data[idx1]['topic_sentence'].append(topic_sentence_cluster_dict)
    
    end = time.time()

    output_3rd = {'clustering' : 
                        {
                        'clustering_tool' : clustering_method,
                        'random_state_number' : random_state_option,
                        'number_of_clusters' : output_2nd.get('clustering').get('number_of_clusters'),
                        'output_centroid_number' : centroid_number_option,
                        'cluster_size_option' : output_2nd.get('clustering').get('cluster_size_option'),
                        'clustering_result' : data
                        },
                     'post_status' : {
                          'error' : None, #violate minimum document number(=30) after deleting duplicated documents
                          'current_documents' : output_2nd.get('post_status').get('current_documents'),
                          'time_1st_module' : output_2nd.get('post_status').get('time_1st_module'),
                          'time_2nd_module' : output_2nd.get('post_status').get('time_2nd_module'),
                          'time_3rd_module' : end - start,
                          'time_total' : output_2nd.get('post_status').get('time_total')+ end - start
              }
        }
    return output_3rd