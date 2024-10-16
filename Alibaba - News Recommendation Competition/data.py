import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.models import YoutubeDNN
from deepmatch.utils import sampledsoftmaxloss,NegativeSampler
warnings.filterwarnings('ignore')

# debug模式：从训练集中划分一些数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
        从训练数据集中随机抽取一部分数据用于调试
        data_path：原数据的存储路径
        sample_nums：采样数目
    """
    all_click=pd.read_csv(data_path+'train_click_log.csv')
    all_user_ids=all_click.user_id.unique()
    
    # 随机抽取sample_nums个数据
    sample_user_ids=np.random.choice(all_user_ids,size=sample_nums,replace=False)
    # 获取sample_user_ids对应的在all_click中的数据
    all_click=all_click[all_click['user_id'].isin(sample_user_ids)]
    
    all_click=all_click.drop_duplicates((['user_id','click_article_id','click_timestamp']))
    return all_click

# 读取点击数据
def get_all_click_df(data_path,offline=True):
    """
        从给定的路径中读取点击数据，并根据offline参数决定是仅读取训练数据还是同时读取训练和测试数据
        data_path：原数据的存储路径
        offline：表示是否处于离线模式。在离线模式下，只处理训练数据，否则，同时处理训练和测试数据
    """
    if offline:
        all_click=pd.read_csv(data_path+'train_click_log.csv')
    else:
        trn_click=pd.read_csv(data_path+'train_click_log.csv')
        tst_click=pd.read_csv(data_path+'testA_click_log.csv')
        
        # 包含测试集和训练集
        all_click=pd.concat([trn_click,tst_click])
    
    # 去除重复的点击记录，保留唯一的(user_id, click_article_id, click_timestamp)组合
    all_click=all_click.drop_duplicates((['user_id','click_article_id','click_timestamp']))
    return all_click

# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df=pd.read_csv(data_path+'articles.csv')
    
    # 为了与训练集中的click_article_id进行拼接，修改article_id为click_article_id
    item_info_df=item_info_df.rename(columns={'article_id':'click_article_id'})
    
    return item_info_df

# 读取文章的embedding属性
def get_item_emb_dict(data_path):
    item_emb_df=pd.read_csv(data_path+'articles_emb.csv')
    # 创建列表item_emb_cols，包含item_emb_df中所有列名包含'emb'的列（用于筛选出包含嵌入向量的列）
    item_emb_cols=[x for x in item_emb_df.columns if 'emb' in x]
    # 利用ascontiguousarray函数将筛选出的嵌入向量列转换为一个连续的Numpy数组item_emb_np
    item_emb_np=np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    item_emb_np=item_emb_np/np.linalg.norm(item_emb_np,axis=1,keepdims=True)
    
    # 创建字典，将item_emb_df中的article_id列的值作为字典的键，将对应的归一化嵌入向量作为字典的值
    item_emb_dict=dict(zip(item_emb_df['article_id'],item_emb_np))
    # 使用pickle库将item_emb_dict字典序列化并保存到文件中
    # wb表示以二进制写入模式打开文件
    save_path='/Users/linjiaxi/Desktop/RecommendationSystem/Competition/Alibaba - News Recommendation Competition/tmp/'
    pickle.dump(item_emb_dict,open(save_path+'item_content_emb.pkl','wb'))
    
    return item_emb_dict


