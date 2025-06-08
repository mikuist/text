# recall_module.py
import collections
import gc
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from lightgbm import early_stopping
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report, confusion_matrix

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_roc_auc(y_true, y_score, title='ROC 曲线'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label='AUC = {:.4f}'.format(auc_val))
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    print('AUC:', auc_val)


def print_classification_metrics(y_true, y_pred, y_score=None):
    print('准确率: {:.4f}'.format(accuracy_score(y_true, y_pred)))
    print('精确率: {:.4f}'.format(precision_score(y_true, y_pred)))
    print('召回率: {:.4f}'.format(recall_score(y_true, y_pred)))
    print('F1 分数: {:.4f}'.format(f1_score(y_true, y_pred)))
    print('\n===分类报告===')
    print(classification_report(y_true, y_pred))
    if y_score is not None:
        auc_val = roc_auc_score(y_true, y_score)
        print('AUC: {:.4f}'.format(auc_val))


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()


def plot_feature_importance(columns, scores, topn=10):
    items = sorted(zip(columns, scores), key=lambda v: v[1], reverse=True)[:topn]
    cols, vals = zip(*items)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(vals), y=list(cols), orient='h')
    plt.xlabel('特征重要性')
    plt.title(f'前 {topn} 个重要特征')
    plt.show()


def plot_feature_importance_pie(columns, scores, topn=10):
    items = sorted(zip(columns, scores), key=lambda v: v[1], reverse=True)[:topn]
    cols, vals = zip(*items)
    plt.figure(figsize=(10, 6))
    plt.pie(vals, labels=cols, autopct='%1.1f%%', startangle=140)
    plt.title(f'前 {topn} 个特征重要性占比')
    plt.show()


def plot_model_comparison(y_true, y_score_1, y_score_2, model1_name='Model 1', model2_name='Model 2'):
    fpr1, tpr1, _ = roc_curve(y_true, y_score_1)
    fpr2, tpr2, _ = roc_curve(y_true, y_score_2)
    auc1 = roc_auc_score(y_true, y_score_1)
    auc2 = roc_auc_score(y_true, y_score_2)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr1, tpr1, label=f'{model1_name} (AUC = {auc1:.4f})')
    plt.plot(fpr2, tpr2, label=f'{model2_name} (AUC = {auc2:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('模型对比ROC曲线')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title='混淆矩阵热图'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()


def plot_prediction_distribution(y_true, y_score, positive_label=1):
    df = pd.DataFrame({'Score': y_score, 'Label': y_true})
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df[df['Label'] == positive_label]['Score'], label='Positive', shade=True, color='blue')
    sns.kdeplot(df[df['Label'] != positive_label]['Score'], label='Negative', shade=True, color='red')
    plt.title('正负样本预测分数分布')
    plt.xlabel('预测分数')
    plt.ylabel('密度')
    plt.legend()
    plt.show()


# ============= 主流程 =============
warnings.filterwarnings('ignore')
np.random.seed(66)
random.seed(66)

data_path = '../data/'
save_path = '../output/'
os.makedirs(save_path, exist_ok=True)

def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {path}, 错误信息: {e}")
        sys.exit(1)

train_click = read_csv_safe(data_path + 'train_click_log.csv')
testA_click = read_csv_safe(data_path + 'testA_click_log.csv')
train_click = pd.concat([train_click, testA_click], axis=0, ignore_index=True)
test_click = read_csv_safe(data_path + 'testB_click_log.csv')
articles = read_csv_safe(data_path + 'articles.csv')
def get_all_click_df(train=True, test=True):
    dfs = []
    if train:
        dfs.append(train_click)
    if test:
        dfs.append(test_click)
    if not dfs:
        print('[ERROR] train和test参数不能同时为False')
        return pd.DataFrame()
    all_click = pd.concat(dfs, ignore_index=True)
    all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    return all_click

def itemcf_recall(topk=10):
    ts = time.time()
    print('ItemCF召回开始...')
    def get_past_click():
        train = train_click.sort_values(['user_id', 'click_timestamp']).reset_index().copy()
        list1, train_indexs = [], []
        for user_id in tqdm(train['user_id'].unique(), ncols=100):
            user = train[train['user_id'] == user_id]
            row = user.tail(1)
            train_indexs.append(row.index.values[0])
            if len(user) >= 2:
                list1.append(row.values.tolist()[0])
        train_last_click = pd.DataFrame(list1, columns=['index', 'user_id', 'article_id', 'click_timestamp',
                                                        'click_environment', 'click_deviceGroup', 'click_os',
                                                        'click_country', 'click_region', 'click_referrer_type'])
        train_last_click = train_last_click.drop(columns=['index'])
        train_past_clicks = train[~train.index.isin(train_indexs)]
        train_past_clicks = train_past_clicks.drop(columns=['index'])
        test = test_click.sort_values(['user_id', 'click_timestamp']).reset_index().copy()
        list2 = []
        for user_id in tqdm(test['user_id'].unique(), ncols=100):
            user = test[test['user_id'] == user_id]
            row = user.tail(1)
            list2.append(row.values.tolist()[0])
        test_last_click = pd.DataFrame(list2, columns=['index', 'user_id', 'article_id', 'click_timestamp',
                                                       'click_environment', 'click_deviceGroup', 'click_os',
                                                       'click_country', 'click_region', 'click_referrer_type'])
        test_last_click = test_last_click.drop(columns=['index'])
        all_click_df = pd.concat([train_past_clicks, test_click], axis=0, ignore_index=True)
        all_click_df = all_click_df.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'], ignore_index=True)
        return all_click_df, train_past_clicks, train_last_click, test_last_click

    def get_user_item_time(click_df):
        click_df = click_df.sort_values('click_timestamp')
        def make_item_time_pair(df):
            return list(zip(df['click_article_id'], df['click_timestamp']))
        user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(
            lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})
        user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
        return user_item_time_dict

    def itemcf_sim(df):
        user_item_time_dict = get_user_item_time(df)
        i2i_sim = {}
        item_cnt = defaultdict(int)
        for user, item_time_list in tqdm(user_item_time_dict.items(), ncols=100):
            for i, i_click_time in item_time_list:
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})
                for j, j_click_time in item_time_list:
                    if i == j:
                        continue
                    i2i_sim[i].setdefault(j, 0)
                    i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
        i2i_sim_ = {}
        for i, related_items in i2i_sim.items():
            i2i_sim_[i] = {}
            for j, wij in related_items.items():
                i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
        pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
        return i2i_sim_

    all_click_df, train_past_clicks, train_last_click, test_last_click = get_past_click()
    i2i_sim = itemcf_sim(all_click_df)

    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
    for i in i2i_sim:
        if not isinstance(i2i_sim[i], list):
            i2i_sim[i] = sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)

    def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num):
        user_hist_items = user_item_time_dict.get(user_id, [])
        user_hist_set = {iid for iid, _ in user_hist_items}
        item_rank = {}
        for loc, (i, click_time) in enumerate(user_hist_items):
            sim_list = i2i_sim.get(i, [])
            for j, wij in sim_list[:sim_item_topk]:
                if j in user_hist_set:
                    continue
                item_rank.setdefault(j, 0)
                item_rank[j] += wij
        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        return item_rank

    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = get_user_item_time(all_click_df)
    sim_item_topk = topk
    recall_item_num = topk
    for user in tqdm(all_click_df['user_id'].unique(), ncols=100):
        user_recall_items_dict[user] = item_based_recommend(
            user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num)

    user_item_score_list = []
    for user, items in tqdm(user_recall_items_dict.items(), ncols=100):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
    recall_df.to_csv(save_path + 'recall_df.csv', index=False)

    tst_recall = recall_df[recall_df['user_id'].isin(test_last_click['user_id'].unique())]
    train_recall = recall_df[recall_df['user_id'].isin(train_last_click['user_id'].unique())]
    test_recall = tst_recall.copy().sort_values(by=['user_id', 'pred_score'], ascending=[True, False])
    train_recall = train_recall.sort_values(by=['user_id', 'pred_score'], ascending=[True, False])
    test_recall = test_recall.drop(columns=['pred_score'], errors='ignore')
    test_recall.to_csv(save_path + 'itemcf_test_recall.csv', index=False)
    train_recall.to_csv(save_path + 'itemcf_train_recall.csv', index=False)
    print('Itemcf Recall Finished! Cost time: {:.2f}s'.format(time.time() - ts))
    return train_past_clicks, train_last_click, test_last_click

def hot_recall(topk=10, train_past_clicks=None, test_last_click=None):
    ts = time.time()
    print('Hot召回开始...')
    train_click_df = get_all_click_df(test=False)
    test_click_df = get_all_click_df(train=False)
    train_click_df = train_click_df.sort_values(['user_id', 'click_timestamp'])
    test_click_df = test_click_df.sort_values(['user_id', 'click_timestamp'])
    articles_copy = articles.rename(columns={'article_id': 'click_article_id'})
    train_click_df = train_click_df.merge(articles_copy, on='click_article_id', how='left')
    test_click_df = test_click_df.merge(articles_copy, on='click_article_id', how='left')
    train_last_click = train_past_clicks.groupby('user_id').agg({'click_timestamp': 'max'}).reset_index()
    train_last_click_time = train_last_click.set_index('user_id')['click_timestamp'].to_dict()
    test_last_click_time = test_last_click.set_index('user_id')['click_timestamp'].to_dict()

    def get_item_topk_click_(hot_articles, hot_articles_dict, click_time, past_click_articles, k):
        topk_click = []
        min_time = click_time - 24 * 60 * 60 * 1000
        max_time = click_time + 24 * 60 * 60 * 1000
        hot_articles_window = hot_articles[(hot_articles['created_at_ts'] >= min_time) & (hot_articles['created_at_ts'] <= max_time)]
        for article_id in hot_articles_window['article_id']:
            if article_id in past_click_articles:
                continue
            topk_click.append(article_id)
            if len(topk_click) == k:
                break
        return topk_click

    train_hot_articles = pd.DataFrame(train_click_df['click_article_id'].value_counts().index.to_list(), columns=['article_id'])
    train_hot_articles = train_hot_articles.merge(articles, on='article_id', how='left').drop(columns=['category_id', 'words_count'], errors='ignore')
    train_hot_articles_dict = train_hot_articles.set_index('article_id')['created_at_ts'].to_dict()
    test_hot_articles = pd.DataFrame(test_click_df['click_article_id'].value_counts().index.to_list(), columns=['article_id'])
    test_hot_articles = test_hot_articles.merge(articles, on='article_id', how='left').drop(columns=['category_id', 'words_count'], errors='ignore')
    test_hot_articles_dict = test_hot_articles.set_index('article_id')['created_at_ts'].to_dict()

    train_list = []
    for user_id in tqdm(train_past_clicks['user_id'].unique()):
        user = train_past_clicks.loc[train_past_clicks['user_id'] == user_id]
        click_time = train_last_click_time[user_id]
        past_click_articles = user['click_article_id'].values
        item_topk_click = get_item_topk_click_(train_hot_articles, train_hot_articles_dict, click_time, past_click_articles, k=topk)
        for aid in item_topk_click:
            train_list.append([user_id, aid])

    hot_train_recall = pd.DataFrame(train_list, columns=['user_id', 'article_id'])
    hot_train_recall.to_csv(save_path + 'hot_train_recall.csv', index=False)

    test_list = []
    for user_id in tqdm(test_click_df['user_id'].unique()):
        user = test_click_df.loc[test_click_df['user_id'] == user_id]
        click_time = test_last_click_time.get(user_id, user['click_timestamp'].max())
        past_click_articles = user['click_article_id'].values
        item_topk_click = get_item_topk_click_(test_hot_articles, test_hot_articles_dict, click_time, past_click_articles, k=topk)
        for aid in item_topk_click:
            test_list.append([user_id, aid])

    hot_test_recall = pd.DataFrame(test_list, columns=['user_id', 'article_id'])
    hot_test_recall.to_csv(save_path + 'hot_test_recall.csv', index=False)
    print('Hot Recall Finished! Cost time: {:.2f}s'.format(time.time() - ts))

def get_test_recall(itemcf=False, hot=False):
    dfs = []
    if itemcf:
        df = pd.read_csv(save_path + 'itemcf_test_recall.csv')
        df = df.rename(columns={'click_article_id': 'article_id'})
        dfs.append(df)
    if hot:
        dfs.append(pd.read_csv(save_path + 'hot_test_recall.csv'))
    if not dfs:
        raise ValueError("至少需要开启一种召回方式！")
    test_recall = pd.concat(dfs, axis=0, ignore_index=True)
    test_recall = test_recall.drop_duplicates(['user_id', 'article_id'])
    test_recall.to_csv(save_path + 'test_recall.csv', index=False)
    print('Test Recall Finished!')
    return test_recall

def get_train_recall(itemcf=False, hot=False, train_last_click=None):
    dfs = []
    if itemcf:
        df = pd.read_csv(save_path + 'itemcf_train_recall.csv')
        df = df.rename(columns={'click_article_id': 'article_id'})
        df = df.merge(train_last_click, on=['user_id', 'article_id'], how='left')
        df['label'] = df['click_timestamp'].apply(lambda x: float(not pd.isna(x)))
        print('Train ItemCF RECALL:{:.2f}%'.format(df['label'].sum() / train_last_click['user_id'].nunique() * 100))
        dfs.append(df)
    if hot:
        df = pd.read_csv(save_path + 'hot_train_recall.csv')
        df['label'] = df.merge(train_last_click, on=['user_id', 'article_id'], how='left')['click_timestamp'].apply(lambda x: float(not pd.isna(x)))
        print('Train Hot RECALL:{:.2f}%'.format(df['label'].sum() / train_last_click['user_id'].nunique() * 100))
        dfs.append(df)
    if not dfs:
        raise ValueError("至少需要开启一种召回方式！")
    train = pd.concat(dfs, axis=0, ignore_index=True)
    train = train.drop_duplicates(['user_id', 'article_id'])
    train['pred_score'] = train.get('pred_score', -100)
    print('Train Total RECALL:{:.2f}%'.format(train['label'].sum() / train_last_click['user_id'].nunique() * 100))
    train.to_csv(save_path + 'train_recall.csv', index=False)

    def neg_sample(train=None, sample_rate=0.001):
        ts = time.time()
        pos_data = train[train['label'] == 1]
        neg_data = train[train['label'] == 0]
        print('pos_data_num:', len(pos_data), 'neg_data_num:', len(neg_data), 'pos/neg:', len(pos_data)/max(1,len(neg_data)))
        def neg_sample_func(group_df):
            neg_num = len(group_df)
            sample_num = max(int(neg_num * sample_rate), 1)
            sample_num = min(sample_num, 5)
            return group_df.sample(n=sample_num, replace=False, random_state=66)
        neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
        neg_data_item_sample = neg_data.groupby('article_id', group_keys=False).apply(neg_sample_func)
        neg_data_new = pd.concat([neg_data_user_sample, neg_data_item_sample], axis=0, ignore_index=True)
        neg_data_new = neg_data_new.drop_duplicates(['user_id', 'article_id'], keep='last')
        data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)
        print('Negative Data Sample Finished! Cost time: {:.2f}s'.format(time.time() - ts))
        return data_new
    train = neg_sample(train)
    return train


def train_and_predict(itemcf=False, itemcf_topk=10, hot=False, hot_topk=10, offline=True):
    ts = time.time()
    if itemcf:
        train_past_clicks, train_last_click, test_last_click = itemcf_recall(itemcf_topk)
    if hot:
        hot_recall(hot_topk, train_past_clicks, test_last_click)

    train_past_clicks_agg = train_past_clicks.groupby('user_id').agg({'click_timestamp': 'max'}).reset_index()
    train = get_train_recall(itemcf, hot, train_last_click)
    train_last_click_wo_ts = train_last_click.drop(columns=['article_id', 'click_timestamp'])
    train = train.drop(columns=['click_timestamp'], errors='ignore').merge(
        train_last_click_wo_ts, how='left', on='user_id'
    )
    train = train.merge(articles, on='article_id', how='left')
    train = train.merge(train_past_clicks_agg, on='user_id', how='left')
    train['delta_time'] = train['created_at_ts'] - train['click_timestamp']

    X = train.copy()
    y = train['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
    X_eval, X_off, y_eval, y_off = train_test_split(X_test, y_test, test_size=0.5, random_state=66)

    g_train = X_train.groupby(['user_id'], as_index=False).count()['label'].values
    g_eval = X_eval.groupby(['user_id'], as_index=False).count()['label'].values

    lgb_cols = [
        'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region',
        'click_referrer_type', 'category_id', 'created_at_ts', 'words_count', 'click_timestamp', 'delta_time'
    ]

    for col in lgb_cols:
        for df in [X_train, X_eval, X_off]:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)

    lgb_ranker = lgb.LGBMRanker(
        boosting_type='gbdt',
        num_leaves=31,
        reg_alpha=0.0,
        reg_lambda=1,
        max_depth=-1,
        n_estimators=1000,
        subsample=0.7,
        colsample_bytree=0.7,
        subsample_freq=1,
        learning_rate=0.01,
        min_child_weight=50,
        random_state=66,
        n_jobs=-1
    )

    lgb_ranker.fit(
        X_train[lgb_cols], y_train, group=g_train,
        eval_set=[(X_eval[lgb_cols], y_eval)], eval_group=[g_eval],
        callbacks=[early_stopping(50, verbose=False)]
    )

    # 只打印非零特征重要性
    feats_imp = list(zip(lgb_cols, lgb_ranker.feature_importances_))
    feats_imp_nonzero = [(col, imp) for col, imp in feats_imp if imp > 0]

    print('-------- 特征重要性 (非零) --------')
    for col, imp in sorted(feats_imp_nonzero, key=lambda x: x[1], reverse=True):
        print(f'{col}: {imp}')
    print('--------------------------')

    # 调用外部绘图函数
    from utils import plot_feature_importance, plot_feature_importance_pie, plot_roc_auc, print_classification_metrics, \
        plot_confusion, plot_model_comparison, plot_prediction_distribution, plot_confusion_matrix

    plot_feature_importance([f[0] for f in feats_imp_nonzero], [f[1] for f in feats_imp_nonzero], topn=10)
    plot_feature_importance_pie([f[0] for f in feats_imp_nonzero], [f[1] for f in feats_imp_nonzero], topn=10)

    X_off['pred_score'] = lgb_ranker.predict(X_off[lgb_cols], num_iteration=lgb_ranker.best_iteration_)

    drop_cols = ['category_id', 'created_at_ts', 'words_count', 'click_environment', 'click_deviceGroup',
                 'click_os', 'click_country', 'click_region', 'click_referrer_type', 'click_timestamp', 'delta_time']

    recall_df = X_off.drop(columns=drop_cols, errors='ignore')

    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'], ascending=[True, False])
    recall_df['rank'] = recall_df.groupby('user_id')['pred_score'].rank(ascending=False, method='first')

    recall_top5 = recall_df[recall_df['rank'] <= 5].copy()

    submit = recall_top5.pivot(index='user_id', columns='rank', values='article_id')
    submit.columns = [f'article_{int(c)}' for c in submit.columns]
    submit = submit.fillna(-1).reset_index()

    submit_file = os.path.join(save_path, f'lgb_ranker_submit_{datetime.today().strftime("%m%d%H%M")}.csv')
    submit.to_csv(submit_file, index=False, sep='\t')
    print(f'提交文件保存至：{submit_file}')

    y_pred = (X_off['pred_score'] > 0.5).astype(int)
    print('\n----- ROC 曲线/AUC -----')
    plot_roc_auc(y_off, X_off['pred_score'])
    print('\n----- 混淆矩阵 -----')
    plot_confusion(y_off, y_pred)
    plot_confusion_matrix(y_off, y_pred, title='混淆矩阵热图')
    print('\n----- 分类指标 -----')
    print_classification_metrics(y_off, y_pred, X_off['pred_score'])

    print('\n----- 特征重要性图表 -----')

    baseline_scores = np.random.rand(len(y_off))
    plot_model_comparison(y_off, X_off['pred_score'], baseline_scores,
                          model1_name='LightGBM Ranker', model2_name='Baseline Model')
    plot_prediction_distribution(y_off, X_off['pred_score'])

    if not offline:
        test_recall = get_test_recall(itemcf, hot)
        test_recall = test_recall.merge(test_last_click.drop(columns=['article_id']), on='user_id', how='left')
        test_recall = test_recall.merge(articles, on='article_id', how='left')
        test_recall['delta_time'] = test_recall['created_at_ts'] - test_recall['click_timestamp']

        for col in lgb_cols:
            if col not in test_recall:
                test_recall[col] = 0
            test_recall[col] = test_recall[col].fillna(0)

        test_recall['pred_score'] = lgb_ranker.predict(test_recall[lgb_cols], num_iteration=lgb_ranker.best_iteration_)

        result = test_recall.sort_values(by=['user_id', 'pred_score'], ascending=[True, False])

        drop_cols2 = ['category_id', 'created_at_ts', 'words_count', 'click_environment', 'clickDeviceGroup',
                      'click_os', 'click_country', 'click_region', 'click_referrer_type', 'click_timestamp',
                      'delta_time']

        result = result.drop(columns=drop_cols2, errors='ignore')

        recall_topk = 5
        result['rank'] = result.groupby('user_id')['pred_score'].rank(ascending=False, method='first')
        test_submit = result[result['rank'] <= recall_topk].copy()
        test_submit = test_submit.pivot(index='user_id', columns='rank', values='article_id')
        test_submit.columns = [f'article_{int(c)}' for c in test_submit.columns]
        test_submit = test_submit.fillna(-1).reset_index()
        submit_test_file = os.path.join(save_path,
                                        f'lgb_ranker_test_submit_{datetime.today().strftime("%m%d%H%M")}.csv')
        test_submit.to_csv(submit_test_file, index=False, sep='\t')
        print(f"测试提交文件保存至：{submit_test_file}")
    mrr_score = 0
    for user_id in tqdm(submit['user_id'].unique()):
        user_rowdf = submit.loc[submit['user_id'] == user_id]
        if user_rowdf.empty:
            continue
        user_row = user_rowdf.iloc[0]
        art_id = train_last_click.loc[train_last_click['user_id'] == user_id, 'article_id'].values[0]
        for i in range(1, 6):
            pred_id = user_row.get('article_{}'.format(i), -1)
            if isinstance(pred_id, pd.Series) or isinstance(pred_id, np.ndarray):
                pred_id = pred_id.iloc[0] if hasattr(pred_id, 'iloc') else pred_id[0]
            if pred_id == art_id:
                mrr_score += 1 / i
    print('MRR: {:.5f}'.format(mrr_score / len(submit['user_id'].unique())))
    print('Submit Finished! Cost time: {:.2f}s'.format(time.time() - ts))

    # 返回测试集数据，方便后续使用(如聚类)
    return X_off, y_off


if __name__ == '__main__':
    offline = False
    itemcf = True
    itemcf_topk = 10
    hot = True
    hot_topk = 10
    train_and_predict(itemcf=itemcf, itemcf_topk=itemcf_topk, hot=hot, hot_topk=hot_topk, offline=offline)
