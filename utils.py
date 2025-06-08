# utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_roc_auc(y_true, y_score, title='ROC 曲线'):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
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
