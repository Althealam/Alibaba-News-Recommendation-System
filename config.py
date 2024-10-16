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

# 获取用户-文章-时间函数
# 根据时间获取用户点击的商品序列: {user1:{item1:time1, item2:time2,...}}
def get_user_item_time(click_df):
    
    click_df = click_df.sort_values('click_timestamp')
    
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    
    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    
    return user_item_time_dict


# 获取文章-用户-时间函数
# 根据时间获取商品被点击的用户序列 {item1: {user1:time1, user2:time2, ...}}
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))
    
    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')[['user_id', 'click_timestamp']].apply(lambda x: make_user_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'user_time_list'})
    
    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict


# 获取历史和最后一次点击
# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    """
        返回两个df：一个是用户的历史点击记录（不包括最后一次点击）
                  一个是用户的最后一次点击记录
    """
    all_click=all_click.sort_values(by=['user_id','click_timestamp'])
    click_last_df=all_click.groupby('user_id').tail(1)
    
    def hist_func(user_df):
        """
            获取每个用户的历史点击记录（不包括最后一次点击）
            如果用户只有一次点击记录，则直接返回这个记录
            如果用户有多次点击记录，返回除了最后一次点击之外的所有记录
        """
        if len(user_df)==1:
            return user_df
        else:
            return user_df[:-1]
    
    # 获取历史点击
    click_hist_df=all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)
    
    # click_hist_df：每个用户的历史点击记录
    # click_last_df：包含每个用户的最后一次点击记录
    return click_hist_df, click_last_df

# 获取文章属性
# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段使用
def get_item_info_dict(item_info_df):
    """
        从一个包含文章（或者商品）信息的df中提取每篇文章的特定属性，并将其保存为字典形式
        提取每篇文章的类别id、词数和创建时间，并且将他们保存在字典中
    """
    max_min_scaler=lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
    # 归一化时间
    item_info_df['created_at_ts']=item_info_df[['created_at_ts']].apply(max_min_scaler)
    
    # 将click_article_id作为键，category_id作为值
    item_type_dict=dict(zip(item_info_df['click_article_id'],item_info_df['category_id']))
    # 将click_article_id作为键，words_count作为值
    item_words_dict=dict(zip(item_info_df['click_article_id'],item_info_df['words_count']))
    # 将click_article_id作为键，归一化后的created_at_ts作为值，创建一个字典
    item_created_time_dict=dict(zip(item_info_df['click_article_id'],item_info_df['created_at_ts']))
    
    # item_type_dict：文章ID到类别ID的映射
    # item_words_dict：文章ID到词数的映射
    # item_created_time_dict：文章ID到归一化创建时间的映射
    return item_type_dict, item_words_dict, item_created_time_dict


# 获取用户历史点击的文章信息
def get_user_hist_item_info_dict(all_click):
    
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs=all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict=dict(zip(user_hist_item_typs['user_id'],user_hist_item_typs['category_id']))
    
    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict=all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict=dict(zip(user_hist_item_ids_dict['user_id'],user_hist_item_ids_dict['click_article_id']))
    
    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words=all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict=dict(zip(user_hist_item_words['user_id'],user_hist_item_words['words_count']))
    
    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
    
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)
    
    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))
    
    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict

# 获取点击次数topk个文章
def get_item_topk_click(click_df,k):
    topk_click=click_df['click_article_id'].value_counts().index[:k]
    return topk_click