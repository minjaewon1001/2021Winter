from flask import Flask, Response, request
from flask_cors import CORS
from flask_cors import cross_origin
from flask.helpers import make_response
from flask import request
from flask import jsonify
from flask import render_template
import logging
import json
import time
############################################################################
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
from cluster_tools import *
from modules import *
import sys
############################################################################

#경고 문구 안 뜨게끔
warnings.filterwarnings(action='ignore')

#flask 인스턴스 객체 생성
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
cors = CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app)

#model path 지정
path = os.getcwd()
model_path = path + '/KoSentenceBERT_SKTBERT/output/training_sts'
#model 실행
embedder = SentenceTransformer(model_path)

#title, content, passage_id를 dictionary 형태로 각각 받는다.
#1. cluster하고 centroid를 출력
#2. centroid 별로 content를 출력
@app.route('/api/news_articles_clustering', methods = ['POST'])
@cross_origin()
def news_articles_clustering():
    response = Response()
    '''
    if request.method == 'OPTIONS':
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "POST")
    elif request.method == 'POST':
        response.headers.add("Access-Control-Allow-Origin", "*")
    '''
    input = request.get_json()
    
    if input.get('output_option') == 1 :
        output_1st = first_module(input.get('data'))
        response.set_data(json.dumps(output_1st, ensure_ascii=False))
        print('1')
        return response
    
    elif input.get('output_option') == 2 :
        output_1st = first_module(input.get('data'))
        output_2nd = second_module(output_1st, input.get('cluster_option_first_module'), embedder)
        response.set_data(json.dumps(output_2nd, ensure_ascii=False))
        print('2')
        return response

    elif input.get('output_option') == 3 :
        output_1st = first_module(input.get('data'))
        output_2nd = second_module(output_1st, input.get('cluster_option_first_module'), embedder)
        if bool(output_2nd.get('post_status').get('error')) == True :
            response.set_data(json.dumps(output_2nd, ensure_ascii=False))
        else :
            output_3rd = third_module(output_2nd, input.get('cluster_option_second_module'), embedder)
            response_weak_depth3(output_3rd)
            response.set_data(json.dumps(output_3rd, ensure_ascii=False))
        print('3')
        print(response)
        return response

def response_weak_depth3(dict_res):
    for i in range(len(dict_res['clustering']['clustering_result'])):
        for j in range(len(dict_res['clustering']['clustering_result'][i]['cluster_data'])):
            dict_res['clustering']['clustering_result'][i]['cluster_data'][j].pop('split_sentence',None)
            dict_res['clustering']['clustering_result'][i]['cluster_data'][j].pop('candidate_sentence',None)


    return 1
if __name__ == '__main__':
    port = sys.argv[1]
    logging.getLogger('flask_cors').level = logging.DEBUG
    app.run(host="10.1.61.200", port = port, debug=False, use_reloader=False, threaded=False)
