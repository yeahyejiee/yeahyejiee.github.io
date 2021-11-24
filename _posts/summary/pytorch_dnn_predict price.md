## pytorch

이 코드는 캐글에서 유명한 집값을 예측하는 것이다.  
파이토치와 탠서플로우의 코드를 비교하고자 함께 진행한다.

### 데이터 다운


```python
import hashlib
import os
import tarfile
import zipfile
import requests

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```


```python
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
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
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```


```python
def download_extract(name, folder=None):  #@save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)
```


```python
%matplotlib inline
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
```


```python
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```


```python
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```


```python
print(train_data.shape)
print(test_data.shape)
```

    (1460, 81)
    (1459, 80)
    


```python
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

       Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice
    0   1          60       RL         65.0       WD        Normal     208500
    1   2          20       RL         80.0       WD        Normal     181500
    2   3          60       RL         68.0       WD        Normal     223500
    3   4          70       RL         60.0       WD       Abnorml     140000
    

## 데이터 묶어서 한번에 하기


```python
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 전처리
- 결측치는 0으로 넣기
- 정규화하기
- 범주형 더미만들기


```python
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```


```python
all_features.shape
```




    (2919, 79)




```python
all_features = pd.get_dummies(all_features, dummy_na=True)  # Dummy_na = True -> considers "na" 
all_features.shape
```




    (2919, 331)



## 데이터 나누기


```python
n_train= train_data.shape[0]   #행
train_features= torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features=torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels= torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)
```

## loss 값정의


```python
loss = nn.MSELoss()   #예측관련된 내용이니 mse를 이용
in_features = train_features.shape[1]  #열

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

def log_rmse(net, features, labels ):  # 안정적인 값을 위해
    clipped_preds=torch.clamp(net(features),1,float('inf'))  #min:1, max= 무한대 에 해당하도록 값을 변경
    rmse=torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()
```


```python
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter=d2l.load_array((train_features, train_labels),batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            l=loss(net(X),y)  #예측과 실제 라벨의 비교를 통한 loss
            l.backward()  # 파라미터에 대한 gradient를 계산 
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls    
```

## train과 valid 데이터 나누기


```python
def get_k_fold_data(k, i, X, y):
    assert k > 1   #가정설정
    fold_size = X.shape[0] // k 
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```


```python
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,batch_size):
    train_l_sum, valid_l_sum = 0, 0
    
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## k개를 통해 나온 결과


```python
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')
```

    fold 1, train log rmse 0.170799, valid log rmse 0.157197
    fold 2, train log rmse 0.162076, valid log rmse 0.187748
    fold 3, train log rmse 0.163779, valid log rmse 0.168360
    fold 4, train log rmse 0.168244, valid log rmse 0.154656
    fold 5, train log rmse 0.163237, valid log rmse 0.183158
    5-fold validation: avg train log rmse: 0.165627, avg valid log rmse: 0.170224
    


    
![svg](output_25_1.svg)
    


## 예측한 결과


```python
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,  #None: TEST데이터가 없어서
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    
    preds = net(test_features).detach().numpy()
    
    #캐글 제출형식
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    #submission.to_csv('submission.csv', index=False)
```


```python
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
```

    train log rmse 0.162424
    


    
![svg](output_28_1.svg)
    



```python
test_data['SalePrice']
```




    0       119365.367188
    1       154094.031250
    2       198728.859375
    3       217300.890625
    4       177165.921875
                ...      
    1454     74520.234375
    1455     85802.257812
    1456    208496.859375
    1457    107149.476562
    1458    240647.984375
    Name: SalePrice, Length: 1459, dtype: float32


