
#중복제거 + 전처리 module

#title, content, embedding을 input으로 받고
#title 기준으로 중복제거 된 title, content, embedding을 산출하고
#title, content는 전처리까지 해서 DataFrame으로 반환, embedding은 그대로 리스트로 반환
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import os
from nltk.tokenize import sent_tokenize
from scipy import spatial
import copy
import json

#text 전처리 module (초기 버전)
def delete_symbols(string):
    #email 제거
    email_pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' 
    processed_string = re.sub(pattern=email_pattern, repl='', string=string)
    #url 제거
    url_pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    processed_string = re.sub(pattern=url_pattern, repl='', string=processed_string)
    #특수문자
    processed_string = re.sub(' +', ' ', re.sub('[^A-Za-z0-9가-힣.?!%]', ' ', string))
    if ' ' == processed_string[0]:
        processed_string = processed_string[1:]
    if ' ' == processed_string[-1]:
        processed_string = processed_string[:-1]

#text list에서 20글자 이하인 것을 제외하고 반환.
def remove_text_under20_in_list(text_list) :
    new_list = []
    for i in text_list :
        if len(i) >= 20 :
            new_list.append(i)
    return new_list

#text_list에서 해당하는 index list만 추출하여 반환
def filter_idx_list(text_list, idx_list) :
    new_text_list = []
    for i in idx_list :
        new_text_list.append(text_list[i])
    return new_text_list
    

#text list에서 20글자 이상인 index list를 반환
def return_text_under20_in_list(text_list) :
    idx_list = []
    for idx, i in enumerate(text_list) :
        if len(i) >= 20 :
            idx_list.append(idx)
    return idx_list

#text 전처리 모듈 (현재 버전)
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
    #pattern = '[^\w\s]'   
    #text = re.sub(pattern=pattern, repl='', string=text)
    #print("특수기호 제거 : ", text , "\n" )
    text = text.strip()
    #print("양 끝 공백 제거 : ", text , "\n" )
    text = " ".join(text.split())
    #print("중간에 공백은 1개만 : ", text )
    return text   
    
def sbert_embedding(titles, contents, embedder) :
    ti_con = []
    for idx, i in enumerate(titles) :
        ti_con.append(i + " " + contents[idx])
    embeddings = embedder.encode(titles)
    embeddings_co = embedder.encode(ti_con)
    return embeddings, embeddings_co

def sbert_titles_embedding(titles, embedder) :
    embeddings = embedder.encode(titles)
    return embeddings

def sbert_titles_embedding_tolist(titles, embedder) :
    embeddings = embedder.encode(titles).tolist()
    return embeddings

def raw2data(temp_titles, temp_contents, embedder) :

    titles_original, contents_original = delete_duplication(temp_titles, temp_contents)
    #titles_data 전처리, titles_embedding data 생성
    titles_processed = []
    for i in titles_original:
        titles_processed.append(clean_text(i))
    titles_embeddings = sbert_titles_embedding_tolist(titles_processed, embedder)

    #contents_original을 전처리 and separating 작업 (clean_text 함수로 작업 시 len이 0이 될 수 있음.)
    #len0 되는 경우는 embedding 할 때 처리 해 주었음.
    contents_split_original = []
    for i in contents_original:
        contents_split_original.append(sent_tokenize(i))
    contents_split_processed = []
    for i in range(len(contents_split_original)) :
        contents_split_processed.append([])
        for j in range(len(contents_split_original[i])) :
            contents_split_processed[i].append([])
            contents_split_processed[i][j] = clean_text(contents_split_original[i][j])


    return titles_original, titles_processed, titles_embeddings, contents_original, contents_split_original, \
                contents_split_processed

def delete_duplication(titles, contents) :
    #original title list, content list 생성
    titles_original = []
    index = []
    n = 0
    for i in titles:
        index.append(n)
        if i not in titles_original:
            titles_original.append(i)  
        else :
            index = index[:-1]
        n += 1
    contents_original = []
    for i in range(len(contents)):
        if i in index:
            contents_original.append(contents[i])
    return titles_original, contents_original

def data2cluster(temp_titles, temp_contents, embedder, target_cluster, num_clusters, random_state) :
    titles_original, contents_original = delete_duplication(temp_titles, temp_contents)
    #titles_data 전처리, titles_embedding data 생성
    titles_processed = []
    for i in titles_original:
        titles_processed.append(clean_text(i))
    titles_embeddings = sbert_titles_embedding_tolist(titles_processed, embedder)

    #contents_original을 전처리 and separating 작업 (clean_text 함수로 작업 시 len이 0이 될 수 있음.)
    #len0 되는 경우는 embedding 할 때 처리 해 주었음.
    contents_split_original = []
    for i in contents_original:
        contents_split_original.append(sent_tokenize(i))
    contents_split_processed = []
    for i in range(len(contents_split_original)) :
        contents_split_processed.append([])
        for j in range(len(contents_split_original[i])) :
            contents_split_processed[i].append([])
            contents_split_processed[i][j] = clean_text(contents_split_original[i][j])


    # clustering 진행    
    target_cluster = target_cluster
    num_clusters = num_clusters
    random_state = random_state
    clustering_model = KMeans(n_clusters=num_clusters,random_state=random_state)
    clustering_model.fit(titles_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(titles_original[sentence_id])
    centroids = clustering_model.cluster_centers_

    #유의 : target_cluster에 대해서만 수행하면 됨!!!
    distances = []
    min_id = []
    target_index = []
    #distance list 만들기, target index의 해당 cluster의 embedding id
    for i in range(len(titles_embeddings)):
        if cluster_assignment[i] == target_cluster-1 :
            target_index.append(i)
            distances.append(np.linalg.norm(centroids[target_cluster - 1] - titles_embeddings[i]))

    #④target index에 따라 titles, contents, separate_context 모두 처리.
    target_titles_embeddings = []
    target_titles_original = []
    target_titles_processed = []
    target_contents_original = []
    target_contents_split_original = []
    target_contents_split_processed = []
    target_contents_split_embeddings = []
    for i in target_index:
        target_titles_embeddings.append(titles_embeddings[i])
        target_titles_original.append(titles_original[i])
        target_titles_processed.append(titles_processed[i])
        target_contents_original.append(contents_original[i])
        target_contents_split_original.append(contents_split_original[i])
        target_contents_split_processed.append(contents_split_processed[i])
    for idx1, i in enumerate(target_contents_split_processed) :
        len_zero_list2 = []
        for idx2, j in enumerate(i) :
            if len(j) == 0 :
                len_zero_list2.append(idx2)
        for j in len_zero_list2 :
            target_contents_split_processed[idx1][j] = "오류 피하기 위해 작성"
        target_contents_split_embeddings.append(sbert_titles_embedding_tolist(i, embedder))
        for j in len_zero_list2 :
            target_contents_split_processed[idx1][j] = None
            target_contents_split_embeddings[idx1][j] = None

    ## centroid data 만들기
    #정렬해서 가장 min한 3개를 centroid로
    temp_i = sorted(distances)
    order = []
    #작은 수 3개 cluster 순서 몇 번째에 있는지 기억
    for j in temp_i[0:3]:
        order.append(distances.index(j))
    for j in order:
        min_id.append(titles_original.index(clustered_sentences[target_cluster - 1][j]))
    centroid_titles_original = []
    centroid_titles_processed = []
    centroid_titles_embeddings = []
    #③ centroid titles 3개 list로... 
    for i in range(len(min_id)):
        centroid_titles_original.append(titles_original[min_id[i]])
        centroid_titles_processed.append(titles_processed[min_id[i]])
        centroid_titles_embeddings.append(titles_embeddings[min_id[i]])

    #similarity between titles and centroids
    titles_similarities = []
    for idx, i in enumerate(target_titles_embeddings) :
        titles_similarities.append([])
        for j in centroid_titles_embeddings :
            if i != None :
                titles_similarities[idx].append(spatial.distance.cosine(i, j))
            else : 
                titles_similarities[idx].append(None)
    #similarity between contents and centroids
    contents_split_similarities = []
    for content_idx, i in enumerate(target_contents_split_embeddings) :
        contents_split_similarities.append([])
        for sentence_idx, j in enumerate(i) :
            contents_split_similarities[content_idx].append([])
            for k in centroid_titles_embeddings :
                if j != None :
                    contents_split_similarities[content_idx][sentence_idx].append(spatial.distance.cosine(j, k))
                else :
                    contents_split_similarities[content_idx][sentence_idx].append(None)

    return target_titles_original, target_titles_processed, target_titles_embeddings, titles_similarities, \
            target_contents_original, \
                target_contents_split_original, target_contents_split_processed, target_contents_split_embeddings, \
                    contents_split_similarities, \
                    centroid_titles_original, centroid_titles_processed, centroid_titles_embeddings

def data2json(temp_titles, temp_contents, embedder, target_cluster, num_clusters, random_state, \
                name, description, file_name) :
    #json_file은 append할 json_file을 의미한다.
    #만약 json_file이 없을 경우 새롭게 만드는 형태로 진행.
    titles_original, titles_processed, titles_embeddings, titles_similarities, \
        contents_original, \
            contents_split_original, contents_split_processed, contents_split_embeddings, \
                contents_split_similarities, \
                centroid_titles_original, centroid_titles_processed, centroid_titles_embeddings \
    = data2cluster(temp_titles, temp_contents, embedder, target_cluster, num_clusters, random_state)

    #default dictionary 정의
    title_default = {"title_original" : None, "title_processed" : None, "title_embedding" : None, "title_similarities" : None}
    content_split_default = {"sentence_original" : None, "sentence_processed" : None, "sentence_embedding" : None, "sentence_similarities" : None}
    centroid_default = {'title_original' : None, 'title_processed' : None, 'title_embedding' : None, 'c.rank' : None}
    data_default = {'title' : None, 'contents' : None, 'content_split' : None}
    status_default = {'cluster_number' : None, 'target_cluster' : None, 'cluster_size' : None}
    dict_default = {'name' : None, 'description' : None, 'status' : None, "data" : None, 'centroids' : None}

    #dictionary value 값 부여
    cluster_dict = copy.deepcopy(dict_default)
    cluster_dict['name'] = name
    cluster_dict['description'] = description
    cluster_dict['status'] = copy.deepcopy(status_default)
    cluster_dict['status']['cluster_number'] = num_clusters
    cluster_dict['status']['target_cluster'] = target_cluster
    cluster_size = len(titles_original)
    cluster_dict['status']['cluster_size'] = cluster_size
    data = []
    for i in range(len(titles_original)) :
        data.append(copy.deepcopy(data_default))
    for i in range(len(titles_original)) :
        data[i]['title'] = copy.deepcopy(title_default)
        data[i]['title']['title_original'] = titles_original[i]
        data[i]['title']['title_processed'] = titles_processed[i]
        data[i]['title']['title_embedding'] = titles_embeddings[i]
        data[i]['title']['title_similarities'] = titles_similarities[i]
        data[i]['contents'] = contents_original[i]
        content_split = []
        for j in range(len(contents_split_original[i])) :
            content_split.append(copy.deepcopy(content_split_default))
        for j in range(len(contents_split_original[i])) :
            content_split[j]['sentence_original'] = contents_split_original[i][j]
            content_split[j]['sentence_processed'] = contents_split_processed[i][j]
            content_split[j]['sentence_embedding'] = contents_split_embeddings[i][j]
            content_split[j]['sentence_similarities'] = contents_split_similarities[i][j]
        data[i]['content_split'] = copy.deepcopy(content_split)
    cluster_dict['data'] = copy.deepcopy(data)
    cluster_dict['centroids'] = []
    for i in range(len(centroid_titles_original)) :
        cluster_dict['centroids'].append(copy.deepcopy(centroid_default))
        cluster_dict['centroids'][i]['title_original'] = centroid_titles_original[i]
        cluster_dict['centroids'][i]['title_processed'] = centroid_titles_processed[i]
        cluster_dict['centroids'][i]['title_embedding'] = centroid_titles_embeddings[i]
        cluster_dict['centroids'][i]['c.rank'] = i+1
    
    if os.path.isfile("data/" + str(file_name)) == False :
        # write first file
        with open(os.getcwd()+"/data/" + str(file_name), 'w', encoding='utf-8-sig') as json_file:
            json.dump([cluster_dict], json_file, ensure_ascii=False, indent = 4)
    else :
        # concat to readed file and overvwrite
        with open(os.getcwd()+"/data/" + str(file_name), 'r', encoding='utf-8-sig') as json_file:
            dicts = json.load(json_file)
        dicts += [cluster_dict] # list concatenation operation . .
        with open(os.getcwd()+"/data/" + str(file_name), 'w', encoding='utf-8-sig') as json_file:
            json.dump(dicts, json_file, ensure_ascii=False, indent = 4)

def make_data(titles, contents, embeddings):
    #title 기준 index
    new_titles = []
    index = []
    n = 0
    for i in titles:
        index.append(n)
        if i not in new_titles:
            new_titles.append(i)  
        else :
            index = index[:-1]
        n += 1
    new_contents = []
    new_embeddings = []
    for i in range(len(contents)):
        if i in index:
            new_contents.append(contents[i])
            new_embeddings.append(embeddings[i])

    #특수문자 마침표 제거 + 연속된 공백 제거 (of title)
    
    titles_char = []
    for i in range(len(new_titles)):
        titles_char.append(re.sub(' +', ' ', re.sub('[^A-Za-z0-9가-힣]', ' ', new_titles[i])))
    for i in range(len(titles_char)):
        if ' ' == titles_char[i][0]:
            titles_char[i] = titles_char[i][1:]
        if ' ' == titles_char[i][-1]:
            titles_char[i] = titles_char[i][:-1]

    #특수문자 마침표 제거 + 연속된 공백 제거
    contents_char = []
    for i in range(len(new_contents)):
        contents_char.append(re.sub(' +', ' ', re.sub('[^A-Za-z0-9가-힣]', ' ', new_contents[i])))
    for i in range(len(contents_char)):
        if ' ' == contents_char[i][0]:
            contents_char[i] = contents_char[i][1:]
        if ' ' == contents_char[i][-1]:
            contents_char[i] = contents_char[i][:-1]
    
    final_contents = contents_char
    final_titles = titles_char
    
    temp_embeddings = np.array(new_embeddings)
    
    final_embeddings = temp_embeddings.reshape(len(new_embeddings),768)

    return final_titles, final_contents, final_embeddings



