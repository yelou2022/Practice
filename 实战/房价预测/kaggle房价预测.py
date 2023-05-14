import hashlib
import os
import tarfile
import zipfile
import requests

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', '房价预测/data')):
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}中下载文件{fname}')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(name)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只能解压zip或是tar格式文件'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    for name in DATA_HUB:
        download(name)


# 获得模型net
def get_net():  # nn.Linear(in_features, out_features) 第一个参数表示输入特征数，第二个参数表示输出特征数
    net = nn.Sequential(nn.Linear(in_features, 1))  # 单层的线性回归
    return net


# 使用对数来衡量相对误差
def log_rmse(net, features, labels):
    clipped_pred = torch.clamp(net(features), 1, float('inf'))  # inf表示无穷大，通过clamp操作将模型输出值控制在1到正无穷之间
    rmse = torch.sqrt(loss(torch.log(clipped_pred), torch.log(labels)))
    return rmse.item()


def train(net, train_feature, train_label, test_feature, test_label, num_epochs, lr, w, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_feature, train_label), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=w)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()  # 梯度清0
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_feature, train_label))
        if test_label is not None:
            test_ls.append(log_rmse(net, test_feature, test_label))
    return train_ls, test_ls


# K折交叉验证
def get_k_fold_data(k, i, X, y):  # k即表示数据分为几份，i表示选定第几份作为验证集，剩余k-1分数据作为训练集
    assert k > 1
    fold_size = X.shape[0] // k  # //表示整除 求每一份的大小
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 对数据集进行分割
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part  # 验证集
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, lr, w, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, lr, w, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse',
                     xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        print(f'折{i+1},训练log rmse:{float(train_ls[-1]):f},' f'验证log rmse:{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


# 提交预测
def train_and_pred(train_feature, test_feature, train_label, test_data, num_epochs, lr, w, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_feature, train_label, None, None, num_epochs, lr, w, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse: {float(train_ls[-1]):f}')
    pred = net(test_feature).detach().numpy()
    test_data['Pred_SalePrice'] = pd.Series(pred.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Pred_SalePrice'], train_data['SalePrice']], axis=1)  # axis=1横向拼接
    submission.to_csv('submission.csv', index=False)


DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# print(train_data.shape)  # (1460, 81)
# print(test_data.shape)  # (1459, 80)


# iloc函数根据位置索引来选择数据
# print('训练数据\n', train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])  # 意思是选取[0,4)行 ，前四列与后三列（-3, -2, -1）的数据
# print('测试数据\n', test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])  # 意思是选取[0,4)行 ，前四列与后三列（-3, -2, -1）的数据

# 在此样本数据中第一行为id列，对我们无用，将其去掉；concat（(A, B), axis）axis默认值为0，其作用为将A，B进行纵向拼接，若axis=1则表示横向拼接
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # 经过此操作后，第一列和最后一列都被删去了,[1:-1)表示排除了第0行和第-1行
# all_feature = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]), 1)  # 经过此操作后，第一列和最后一列都被删去了,[1:-1)表示排除了第0行和第-1行
#
# print(all_features.shape)  # (2919, 79)
# print(all_feature.shape)  # (1460, 158)
# =====================================================================
# 清洗数据:将所有缺失的值（NA）替换为相应特征的平均值.通过将特征重新缩放到0均值和单位方差来标准化数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 填充0

# 处理字符串 get_dummies将数据离散化，使得列扩张为331
all_features = pd.get_dummies(all_features, dummy_na=True)  # 使用one-hot编码来处理字符串
# all_features.shape:  (2919,331)

n_train = train_data.shape[0]  # shape的第一个参数为行，此处将train_data的行数赋给n_train， 便于后续分离
# print(n_train)  # 1460
# 之前通过concat进行过纵向拼接，此处进行分离
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)  # 前n_train行为训练数据
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)  # n_train行以后的为测试数据

# 提取销售价格那一列数据，并转换为列向量
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)  # 变成1列，行数自动计算

# 训练
loss = nn.MSELoss()
in_features = train_features.shape[1]
# print(in_features)  # 331


k, num_epochs, lr, w, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, w, batch_size)
print(f'{k}折验证：平均训练log rmse： {float(train_l):f}' f'平均验证log rmse： {float(valid_l):f}')
d2l.plt.show()
# 测试提交
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, w, batch_size)
d2l.plt.show()

# print(f'token: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')
