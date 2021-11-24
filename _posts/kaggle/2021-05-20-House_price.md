---

author: Hone Ye ji
categories: 
 - Kaggle
 - ml
tags: 
 - kaggle
 
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---


kaggle의 House price를 연습해보겠습니다.


### House Price
#### id로 salePrice를 예측


```python
import pandas as pd
```

## 1. data load


```python
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
```

#### 변수 소개
- MSSubClass: 매매와 관련된 주거 유형을 식별
- MSZoning: 판매의 일반 구역 분류를 식별 
  
  (   A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density )
- LotFrontage:재산에 연결된 선으로 된 피트
- LotArea:로트 크기(평방 피트)
- Street:숙박시설에 대한 도로 접근 유형 (Grv:Gravel,Pave:Paved)
- Alley: 골목을 통한 부동산 접근 유형 (Grvl: Gravel,Pave: Paved, NA: No alley access)

등 81개의 변수



```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
train.shape
```




    (1460, 81)




```python
test.shape
```




    (1459, 80)




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    


```python
numeric=train.dtypes[train.dtypes != "object"].index
print("숫자형의 변수 종류:", len(numeric))
      
cate=train.dtypes[train.dtypes == 'object'].index
print("문자형의 변수 종류:",len(cate))

print("=====")
print('숫자형 변수',numeric)
print('문자형 변수',cate)
```

    숫자형의 변수 종류: 38
    문자형의 변수 종류: 43
    =====
    숫자형 변수 Index(['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
           'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
           'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
           'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
           'MiscVal', 'MoSold', 'YrSold', 'SalePrice'],
          dtype='object')
    문자형 변수 Index(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
           'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
           'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
           'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
           'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
           'SaleType', 'SaleCondition'],
          dtype='object')
    

### 1)이상치 탐색 및 제거


```python
import numpy as np
from collections import Counter


def detect_outliers(df, n, features): 
    outlier_indices = [] 
    for col in features: 
        Q1 = np.percentile(df[col], 25) 
        Q3 = np.percentile(df[col], 75) 
        IQR = Q3 - Q1 
        outlier_step = 1.5 * IQR 
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index 
        outlier_indices.extend(outlier_list_col) 
    outlier_indices = Counter(outlier_indices) 
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n) 
        
    return multiple_outliers


```


```python
Outliers_to_drop = detect_outliers(train, 2, 
                                   ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
                                    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 
                                    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                                    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'])

```


```python
train.loc[Outliers_to_drop]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>190</td>
      <td>RM</td>
      <td>33.0</td>
      <td>4456</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2009</td>
      <td>New</td>
      <td>Partial</td>
      <td>113000</td>
    </tr>
    <tr>
      <th>93</th>
      <td>94</td>
      <td>190</td>
      <td>C (all)</td>
      <td>60.0</td>
      <td>7200</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>133900</td>
    </tr>
    <tr>
      <th>125</th>
      <td>126</td>
      <td>190</td>
      <td>RM</td>
      <td>60.0</td>
      <td>6780</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>84500</td>
    </tr>
    <tr>
      <th>165</th>
      <td>166</td>
      <td>190</td>
      <td>RL</td>
      <td>62.0</td>
      <td>10106</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>127500</td>
    </tr>
    <tr>
      <th>172</th>
      <td>173</td>
      <td>160</td>
      <td>RL</td>
      <td>44.0</td>
      <td>5306</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>239000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1350</th>
      <td>1351</td>
      <td>90</td>
      <td>RL</td>
      <td>91.0</td>
      <td>11643</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>200000</td>
    </tr>
    <tr>
      <th>634</th>
      <td>635</td>
      <td>90</td>
      <td>RL</td>
      <td>64.0</td>
      <td>6979</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>600</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>144000</td>
    </tr>
    <tr>
      <th>910</th>
      <td>911</td>
      <td>90</td>
      <td>RL</td>
      <td>80.0</td>
      <td>11600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>154300</td>
    </tr>
    <tr>
      <th>1292</th>
      <td>1293</td>
      <td>70</td>
      <td>RM</td>
      <td>60.0</td>
      <td>6600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>107500</td>
    </tr>
    <tr>
      <th>912</th>
      <td>913</td>
      <td>30</td>
      <td>RM</td>
      <td>51.0</td>
      <td>6120</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>620</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>88000</td>
    </tr>
  </tbody>
</table>
<p>122 rows × 81 columns</p>
</div>



122행에서 이상치 발견


```python
train=train.drop(Outliers_to_drop,axis=0).reset_index(drop=True)
train.shape
```




    (1338, 81)



### 2) 결측치 발견


```python
for col in train.columns:
    msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (train[col].isnull().sum() / train[col].shape[0])) 
    print(msperc)


```

    column:         Id	 Percent of NaN value: 0.00%
    column: MSSubClass	 Percent of NaN value: 0.00%
    column:   MSZoning	 Percent of NaN value: 0.00%
    column: LotFrontage	 Percent of NaN value: 17.12%
    column:    LotArea	 Percent of NaN value: 0.00%
    column:     Street	 Percent of NaN value: 0.00%
    column:      Alley	 Percent of NaN value: 94.10%
    column:   LotShape	 Percent of NaN value: 0.00%
    column: LandContour	 Percent of NaN value: 0.00%
    column:  Utilities	 Percent of NaN value: 0.00%
    column:  LotConfig	 Percent of NaN value: 0.00%
    column:  LandSlope	 Percent of NaN value: 0.00%
    column: Neighborhood	 Percent of NaN value: 0.00%
    column: Condition1	 Percent of NaN value: 0.00%
    column: Condition2	 Percent of NaN value: 0.00%
    column:   BldgType	 Percent of NaN value: 0.00%
    column: HouseStyle	 Percent of NaN value: 0.00%
    column: OverallQual	 Percent of NaN value: 0.00%
    column: OverallCond	 Percent of NaN value: 0.00%
    column:  YearBuilt	 Percent of NaN value: 0.00%
    column: YearRemodAdd	 Percent of NaN value: 0.00%
    column:  RoofStyle	 Percent of NaN value: 0.00%
    column:   RoofMatl	 Percent of NaN value: 0.00%
    column: Exterior1st	 Percent of NaN value: 0.00%
    column: Exterior2nd	 Percent of NaN value: 0.00%
    column: MasVnrType	 Percent of NaN value: 0.52%
    column: MasVnrArea	 Percent of NaN value: 0.52%
    column:  ExterQual	 Percent of NaN value: 0.00%
    column:  ExterCond	 Percent of NaN value: 0.00%
    column: Foundation	 Percent of NaN value: 0.00%
    column:   BsmtQual	 Percent of NaN value: 2.32%
    column:   BsmtCond	 Percent of NaN value: 2.32%
    column: BsmtExposure	 Percent of NaN value: 2.39%
    column: BsmtFinType1	 Percent of NaN value: 2.32%
    column: BsmtFinSF1	 Percent of NaN value: 0.00%
    column: BsmtFinType2	 Percent of NaN value: 2.39%
    column: BsmtFinSF2	 Percent of NaN value: 0.00%
    column:  BsmtUnfSF	 Percent of NaN value: 0.00%
    column: TotalBsmtSF	 Percent of NaN value: 0.00%
    column:    Heating	 Percent of NaN value: 0.00%
    column:  HeatingQC	 Percent of NaN value: 0.00%
    column: CentralAir	 Percent of NaN value: 0.00%
    column: Electrical	 Percent of NaN value: 0.07%
    column:   1stFlrSF	 Percent of NaN value: 0.00%
    column:   2ndFlrSF	 Percent of NaN value: 0.00%
    column: LowQualFinSF	 Percent of NaN value: 0.00%
    column:  GrLivArea	 Percent of NaN value: 0.00%
    column: BsmtFullBath	 Percent of NaN value: 0.00%
    column: BsmtHalfBath	 Percent of NaN value: 0.00%
    column:   FullBath	 Percent of NaN value: 0.00%
    column:   HalfBath	 Percent of NaN value: 0.00%
    column: BedroomAbvGr	 Percent of NaN value: 0.00%
    column: KitchenAbvGr	 Percent of NaN value: 0.00%
    column: KitchenQual	 Percent of NaN value: 0.00%
    column: TotRmsAbvGrd	 Percent of NaN value: 0.00%
    column: Functional	 Percent of NaN value: 0.00%
    column: Fireplaces	 Percent of NaN value: 0.00%
    column: FireplaceQu	 Percent of NaN value: 48.28%
    column: GarageType	 Percent of NaN value: 4.86%
    column: GarageYrBlt	 Percent of NaN value: 4.86%
    column: GarageFinish	 Percent of NaN value: 4.86%
    column: GarageCars	 Percent of NaN value: 0.00%
    column: GarageArea	 Percent of NaN value: 0.00%
    column: GarageQual	 Percent of NaN value: 4.86%
    column: GarageCond	 Percent of NaN value: 4.86%
    column: PavedDrive	 Percent of NaN value: 0.00%
    column: WoodDeckSF	 Percent of NaN value: 0.00%
    column: OpenPorchSF	 Percent of NaN value: 0.00%
    column: EnclosedPorch	 Percent of NaN value: 0.00%
    column:  3SsnPorch	 Percent of NaN value: 0.00%
    column: ScreenPorch	 Percent of NaN value: 0.00%
    column:   PoolArea	 Percent of NaN value: 0.00%
    column:     PoolQC	 Percent of NaN value: 99.85%
    column:      Fence	 Percent of NaN value: 80.94%
    column: MiscFeature	 Percent of NaN value: 97.16%
    column:    MiscVal	 Percent of NaN value: 0.00%
    column:     MoSold	 Percent of NaN value: 0.00%
    column:     YrSold	 Percent of NaN value: 0.00%
    column:   SaleType	 Percent of NaN value: 0.00%
    column: SaleCondition	 Percent of NaN value: 0.00%
    column:  SalePrice	 Percent of NaN value: 0.00%
    


```python
for col in test.columns:
    msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (train[col].isnull().sum() / train[col].shape[0])) 
    print(msperc)
```

    column:         Id	 Percent of NaN value: 0.00%
    column: MSSubClass	 Percent of NaN value: 0.00%
    column:   MSZoning	 Percent of NaN value: 0.00%
    column: LotFrontage	 Percent of NaN value: 17.12%
    column:    LotArea	 Percent of NaN value: 0.00%
    column:     Street	 Percent of NaN value: 0.00%
    column:      Alley	 Percent of NaN value: 94.10%
    column:   LotShape	 Percent of NaN value: 0.00%
    column: LandContour	 Percent of NaN value: 0.00%
    column:  Utilities	 Percent of NaN value: 0.00%
    column:  LotConfig	 Percent of NaN value: 0.00%
    column:  LandSlope	 Percent of NaN value: 0.00%
    column: Neighborhood	 Percent of NaN value: 0.00%
    column: Condition1	 Percent of NaN value: 0.00%
    column: Condition2	 Percent of NaN value: 0.00%
    column:   BldgType	 Percent of NaN value: 0.00%
    column: HouseStyle	 Percent of NaN value: 0.00%
    column: OverallQual	 Percent of NaN value: 0.00%
    column: OverallCond	 Percent of NaN value: 0.00%
    column:  YearBuilt	 Percent of NaN value: 0.00%
    column: YearRemodAdd	 Percent of NaN value: 0.00%
    column:  RoofStyle	 Percent of NaN value: 0.00%
    column:   RoofMatl	 Percent of NaN value: 0.00%
    column: Exterior1st	 Percent of NaN value: 0.00%
    column: Exterior2nd	 Percent of NaN value: 0.00%
    column: MasVnrType	 Percent of NaN value: 0.52%
    column: MasVnrArea	 Percent of NaN value: 0.52%
    column:  ExterQual	 Percent of NaN value: 0.00%
    column:  ExterCond	 Percent of NaN value: 0.00%
    column: Foundation	 Percent of NaN value: 0.00%
    column:   BsmtQual	 Percent of NaN value: 2.32%
    column:   BsmtCond	 Percent of NaN value: 2.32%
    column: BsmtExposure	 Percent of NaN value: 2.39%
    column: BsmtFinType1	 Percent of NaN value: 2.32%
    column: BsmtFinSF1	 Percent of NaN value: 0.00%
    column: BsmtFinType2	 Percent of NaN value: 2.39%
    column: BsmtFinSF2	 Percent of NaN value: 0.00%
    column:  BsmtUnfSF	 Percent of NaN value: 0.00%
    column: TotalBsmtSF	 Percent of NaN value: 0.00%
    column:    Heating	 Percent of NaN value: 0.00%
    column:  HeatingQC	 Percent of NaN value: 0.00%
    column: CentralAir	 Percent of NaN value: 0.00%
    column: Electrical	 Percent of NaN value: 0.07%
    column:   1stFlrSF	 Percent of NaN value: 0.00%
    column:   2ndFlrSF	 Percent of NaN value: 0.00%
    column: LowQualFinSF	 Percent of NaN value: 0.00%
    column:  GrLivArea	 Percent of NaN value: 0.00%
    column: BsmtFullBath	 Percent of NaN value: 0.00%
    column: BsmtHalfBath	 Percent of NaN value: 0.00%
    column:   FullBath	 Percent of NaN value: 0.00%
    column:   HalfBath	 Percent of NaN value: 0.00%
    column: BedroomAbvGr	 Percent of NaN value: 0.00%
    column: KitchenAbvGr	 Percent of NaN value: 0.00%
    column: KitchenQual	 Percent of NaN value: 0.00%
    column: TotRmsAbvGrd	 Percent of NaN value: 0.00%
    column: Functional	 Percent of NaN value: 0.00%
    column: Fireplaces	 Percent of NaN value: 0.00%
    column: FireplaceQu	 Percent of NaN value: 48.28%
    column: GarageType	 Percent of NaN value: 4.86%
    column: GarageYrBlt	 Percent of NaN value: 4.86%
    column: GarageFinish	 Percent of NaN value: 4.86%
    column: GarageCars	 Percent of NaN value: 0.00%
    column: GarageArea	 Percent of NaN value: 0.00%
    column: GarageQual	 Percent of NaN value: 4.86%
    column: GarageCond	 Percent of NaN value: 4.86%
    column: PavedDrive	 Percent of NaN value: 0.00%
    column: WoodDeckSF	 Percent of NaN value: 0.00%
    column: OpenPorchSF	 Percent of NaN value: 0.00%
    column: EnclosedPorch	 Percent of NaN value: 0.00%
    column:  3SsnPorch	 Percent of NaN value: 0.00%
    column: ScreenPorch	 Percent of NaN value: 0.00%
    column:   PoolArea	 Percent of NaN value: 0.00%
    column:     PoolQC	 Percent of NaN value: 99.85%
    column:      Fence	 Percent of NaN value: 80.94%
    column: MiscFeature	 Percent of NaN value: 97.16%
    column:    MiscVal	 Percent of NaN value: 0.00%
    column:     MoSold	 Percent of NaN value: 0.00%
    column:     YrSold	 Percent of NaN value: 0.00%
    column:   SaleType	 Percent of NaN value: 0.00%
    column: SaleCondition	 Percent of NaN value: 0.00%
    

#### 결측치가 1개이상인 변수에 대해 확인


```python
missing = train.isnull().sum() 
missing = missing[missing > 0] 
missing.sort_values(inplace=True) 
missing.plot.bar(figsize = (12,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x202a2994490>




![output_21_1](https://user-images.githubusercontent.com/45659433/119081337-bf92c780-ba36-11eb-8cd7-55abee47add2.png)



```python
missing = test.isnull().sum() 
missing = missing[missing > 0] 
missing.sort_values(inplace=True) 
missing.plot.bar(figsize = (12,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x202a312b610>




![output_22_1](https://user-images.githubusercontent.com/45659433/119081383-d5a08800-ba36-11eb-9663-990665b09a22.png)

LotFrontage, Fence, Alley, MiscFeature, PoolQC 의 변수들에서 결측치가 많이 발견

### 3) Skewness (왜도=비대칭도)
- 왜도 : 얼마나 비대칭인지 , 정규분포가 아닌지 , a<0 오른쪽에 치우침,
- 첨도: 뾰족한 정도 , 중심에 집중되어 있는지


```python
for col in numeric:
    print('{:15}'.format(col), 'Skewness: {:05.2f}'.format(train[col].skew()) , ' ' , 'Kurtosis: {:06.2f}'.format(train[col].kurt()) )

```

    Id              Skewness: 00.00   Kurtosis: -01.19
    MSSubClass      Skewness: 01.37   Kurtosis: 001.49
    LotFrontage     Skewness: 01.59   Kurtosis: 013.04
    LotArea         Skewness: 07.78   Kurtosis: 123.55
    OverallQual     Skewness: 00.16   Kurtosis: -00.05
    OverallCond     Skewness: 00.74   Kurtosis: 001.23
    YearBuilt       Skewness: -0.58   Kurtosis: -00.60
    YearRemodAdd    Skewness: -0.52   Kurtosis: -01.27
    MasVnrArea      Skewness: 02.69   Kurtosis: 011.04
    BsmtFinSF1      Skewness: 00.65   Kurtosis: -00.50
    BsmtFinSF2      Skewness: 04.86   Kurtosis: 026.86
    BsmtUnfSF       Skewness: 00.87   Kurtosis: 000.29
    TotalBsmtSF     Skewness: 00.29   Kurtosis: 001.17
    1stFlrSF        Skewness: 00.66   Kurtosis: 000.02
    2ndFlrSF        Skewness: 00.77   Kurtosis: -00.80
    LowQualFinSF    Skewness: 12.74   Kurtosis: 170.50
    GrLivArea       Skewness: 00.66   Kurtosis: 000.54
    BsmtFullBath    Skewness: 00.56   Kurtosis: -00.96
    BsmtHalfBath    Skewness: 04.91   Kurtosis: 024.68
    FullBath        Skewness: -0.05   Kurtosis: -01.25
    HalfBath        Skewness: 00.66   Kurtosis: -01.14
    BedroomAbvGr    Skewness: -0.16   Kurtosis: 000.92
    KitchenAbvGr    Skewness: 05.36   Kurtosis: 032.14
    TotRmsAbvGrd    Skewness: 00.37   Kurtosis: 000.11
    Fireplaces      Skewness: 00.63   Kurtosis: -00.28
    GarageYrBlt     Skewness: -0.64   Kurtosis: -00.51
    GarageCars      Skewness: -0.35   Kurtosis: 000.20
    GarageArea      Skewness: 00.07   Kurtosis: 000.62
    WoodDeckSF      Skewness: 01.35   Kurtosis: 001.93
    OpenPorchSF     Skewness: 02.08   Kurtosis: 006.50
    EnclosedPorch   Skewness: 03.00   Kurtosis: 008.34
    3SsnPorch       Skewness: 10.92   Kurtosis: 138.82
    ScreenPorch     Skewness: 04.23   Kurtosis: 018.48
    PoolArea        Skewness: 25.97   Kurtosis: 675.73
    MiscVal         Skewness: 31.39   Kurtosis: 1075.77
    MoSold          Skewness: 00.22   Kurtosis: -00.42
    YrSold          Skewness: 00.10   Kurtosis: -01.19
    SalePrice       Skewness: 01.29   Kurtosis: 002.56
    

## 2. Numerical Feature


```python
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from collections import Counter 
plt.style.use('seaborn') 
sns.set(font_scale=1.5) 
#import missingno as msno 
import warnings 
warnings.filterwarnings("ignore") 

%matplotlib inline
```

#### numeric한 변수에 대해서 만

- 변수의 상관관계를 파악 
- 색이 진하면 관계가 높다 (다중공선성이 있을 수 있다)


```python
corr_data = train[['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
                     'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                     'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                     'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
                     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']] 

colormap = plt.cm.PuBu 
sns.set(font_scale=1.0) 

f , ax = plt.subplots(figsize = (14,12)) 
plt.title('Correlation of Numeric Features with Sale Price',y=1,size=18) 
sns.heatmap(corr_data.corr(),square = True, linewidths = 0.1, cmap = colormap, linecolor = "white", vmax=0.8)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x202a55d0250>



![output_30_1](https://user-images.githubusercontent.com/45659433/119081384-d6391e80-ba36-11eb-8ef9-c2512a361e8d.png)



```python
k= 11 
cols = corr_data.corr().nlargest(k,'SalePrice')['SalePrice'].index 
print(cols) 
cm = np.corrcoef(train[cols].values.T) 

f , ax = plt.subplots(figsize = (12,10)) 
sns.heatmap(cm, vmax=.8, linewidths=0.1,square=True,annot=True,cmap=colormap, 
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':14},yticklabels = cols.values)

```

    Index(['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
           'TotalBsmtSF', '1stFlrSF', 'YearBuilt', 'FullBath', 'TotRmsAbvGrd',
           'YearRemodAdd'],
          dtype='object')
    




    <matplotlib.axes._subplots.AxesSubplot at 0x202a5a59250>



![output_31_2](https://user-images.githubusercontent.com/45659433/119081385-d6391e80-ba36-11eb-9948-96635e7223cd.png)


- GarageCars와 GarageArea =0.88


```python
sns.set() 
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars','FullBath','YearBuilt','YearRemodAdd'] 
sns.pairplot(train[columns],size = 2 ,kind ='scatter',diag_kind='kde') 
plt.show()


```


![output_33_0](https://user-images.githubusercontent.com/45659433/119081388-d6d1b500-ba36-11eb-84a2-278db4198c4d.png)



```python
fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(16,13)) 
OverallQual_scatter_plot = pd.concat([train['SalePrice'],train['OverallQual']],axis = 1) 
sns.regplot(x='OverallQual',y = 'SalePrice',data = OverallQual_scatter_plot,scatter= True, fit_reg=True, ax=ax1) 
TotalBsmtSF_scatter_plot = pd.concat([train['SalePrice'],train['TotalBsmtSF']],axis = 1) 
sns.regplot(x='TotalBsmtSF',y = 'SalePrice',data = TotalBsmtSF_scatter_plot,scatter= True, fit_reg=True, ax=ax2) 
GrLivArea_scatter_plot = pd.concat([train['SalePrice'],train['GrLivArea']],axis = 1) 
sns.regplot(x='GrLivArea',y = 'SalePrice',data = GrLivArea_scatter_plot,scatter= True, fit_reg=True, ax=ax3) 
GarageCars_scatter_plot = pd.concat([train['SalePrice'],train['GarageCars']],axis = 1) 
sns.regplot(x='GarageCars',y = 'SalePrice',data = GarageCars_scatter_plot,scatter= True, fit_reg=True, ax=ax4) 
FullBath_scatter_plot = pd.concat([train['SalePrice'],train['FullBath']],axis = 1) 
sns.regplot(x='FullBath',y = 'SalePrice',data = FullBath_scatter_plot,scatter= True, fit_reg=True, ax=ax5) 
YearBuilt_scatter_plot = pd.concat([train['SalePrice'],train['YearBuilt']],axis = 1) 
sns.regplot(x='YearBuilt',y = 'SalePrice',data = YearBuilt_scatter_plot,scatter= True, fit_reg=True, ax=ax6) 
YearRemodAdd_scatter_plot = pd.concat([train['SalePrice'],train['YearRemodAdd']],axis = 1) 
YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd','SalePrice')


```

    *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2-D array with a single row if you intend to specify the same RGB or RGBA value for all points.
    




    <matplotlib.axes._subplots.AxesSubplot at 0x202a9e3ceb0>




![output_34_2](https://user-images.githubusercontent.com/45659433/119081390-d6d1b500-ba36-11eb-9410-4c993af132ef.png)



![output_34_3](https://user-images.githubusercontent.com/45659433/119081380-d507f180-ba36-11eb-97e9-46f434d8269f.png)


- OverallQual, GarageCars, Fullbath 는 범주형 특성 (등급, 갯수)

### categorical 변수


```python
for catg in list(cate) :
    print(train[catg].value_counts()) 
    print('#'*50)


```

    RL         1055
    RM          197
    FV           65
    RH           14
    C (all)       7
    Name: MSZoning, dtype: int64
    ##################################################
    Pave    1335
    Grvl       3
    Name: Street, dtype: int64
    ##################################################
    Grvl    42
    Pave    37
    Name: Alley, dtype: int64
    ##################################################
    Reg    859
    IR1    440
    IR2     32
    IR3      7
    Name: LotShape, dtype: int64
    ##################################################
    Lvl    1211
    Bnk      52
    HLS      48
    Low      27
    Name: LandContour, dtype: int64
    ##################################################
    AllPub    1338
    Name: Utilities, dtype: int64
    ##################################################
    Inside     972
    Corner     235
    CulDSac     81
    FR2         46
    FR3          4
    Name: LotConfig, dtype: int64
    ##################################################
    Gtl    1275
    Mod      56
    Sev       7
    Name: LandSlope, dtype: int64
    ##################################################
    NAmes      205
    CollgCr    149
    OldTown     93
    Edwards     92
    Somerst     85
    Gilbert     79
    NridgHt     71
    Sawyer      68
    NWAmes      66
    BrkSide     56
    SawyerW     54
    Mitchel     46
    Crawfor     44
    NoRidge     37
    IDOTRR      33
    Timber      32
    StoneBr     21
    SWISU       20
    ClearCr     19
    Blmngtn     17
    BrDale      16
    MeadowV     15
    Veenker     10
    NPkVill      8
    Blueste      2
    Name: Neighborhood, dtype: int64
    ##################################################
    Norm      1165
    Feedr       70
    Artery      42
    RRAn        24
    PosN        12
    RRAe        11
    PosA         7
    RRNn         5
    RRNe         2
    Name: Condition1, dtype: int64
    ##################################################
    Norm      1327
    Feedr        5
    RRNn         2
    Artery       2
    RRAn         1
    PosN         1
    Name: Condition2, dtype: int64
    ##################################################
    1Fam      1128
    TwnhsE     110
    Duplex      43
    Twnhs       43
    2fmCon      14
    Name: BldgType, dtype: int64
    ##################################################
    1Story    681
    2Story    402
    1.5Fin    138
    SLvl       58
    SFoyer     36
    1.5Unf     14
    2.5Unf      8
    2.5Fin      1
    Name: HouseStyle, dtype: int64
    ##################################################
    Gable      1067
    Hip         246
    Gambrel      11
    Flat          8
    Mansard       5
    Shed          1
    Name: RoofStyle, dtype: int64
    ##################################################
    CompShg    1324
    Tar&Grv       6
    WdShake       3
    WdShngl       2
    Roll          1
    Membran       1
    Metal         1
    Name: RoofMatl, dtype: int64
    ##################################################
    VinylSd    493
    HdBoard    211
    MetalSd    202
    Wd Sdng    176
    Plywood     92
    CemntBd     54
    BrkFace     41
    WdShing     24
    Stucco      21
    AsbShng     18
    BrkComm      2
    CBlock       1
    ImStucc      1
    Stone        1
    AsphShn      1
    Name: Exterior1st, dtype: int64
    ##################################################
    VinylSd    484
    MetalSd    196
    HdBoard    192
    Wd Sdng    173
    Plywood    124
    CmentBd     54
    Wd Shng     34
    Stucco      21
    BrkFace     19
    AsbShng     18
    ImStucc      8
    Brk Cmn      6
    Stone        4
    AsphShn      3
    CBlock       1
    Other        1
    Name: Exterior2nd, dtype: int64
    ##################################################
    None       786
    BrkFace    417
    Stone      117
    BrkCmn      11
    Name: MasVnrType, dtype: int64
    ##################################################
    TA    833
    Gd    459
    Ex     38
    Fa      8
    Name: ExterQual, dtype: int64
    ##################################################
    TA    1192
    Gd     122
    Fa      22
    Po       1
    Ex       1
    Name: ExterCond, dtype: int64
    ##################################################
    PConc     614
    CBlock    571
    BrkTil    128
    Slab       19
    Stone       3
    Wood        3
    Name: Foundation, dtype: int64
    ##################################################
    Gd    588
    TA    583
    Ex    104
    Fa     32
    Name: BsmtQual, dtype: int64
    ##################################################
    TA    1211
    Gd      57
    Fa      37
    Po       2
    Name: BsmtCond, dtype: int64
    ##################################################
    No    881
    Av    209
    Gd    113
    Mn    103
    Name: BsmtExposure, dtype: int64
    ##################################################
    Unf    401
    GLQ    388
    ALQ    197
    BLQ    137
    Rec    119
    LwQ     65
    Name: BsmtFinType1, dtype: int64
    ##################################################
    Unf    1183
    LwQ      39
    Rec      35
    BLQ      23
    ALQ      15
    GLQ      11
    Name: BsmtFinType2, dtype: int64
    ##################################################
    GasA     1313
    GasW       13
    Grav        7
    Wall        3
    Floor       1
    OthW        1
    Name: Heating, dtype: int64
    ##################################################
    Ex    690
    TA    390
    Gd    215
    Fa     42
    Po      1
    Name: HeatingQC, dtype: int64
    ##################################################
    Y    1263
    N      75
    Name: CentralAir, dtype: int64
    ##################################################
    SBrkr    1223
    FuseA      88
    FuseF      23
    FuseP       2
    Mix         1
    Name: Electrical, dtype: int64
    ##################################################
    TA    669
    Gd    555
    Ex     81
    Fa     33
    Name: KitchenQual, dtype: int64
    ##################################################
    Typ     1258
    Min2      28
    Min1      25
    Mod       11
    Maj1      11
    Maj2       5
    Name: Functional, dtype: int64
    ##################################################
    Gd    344
    TA    282
    Fa     32
    Ex     18
    Po     16
    Name: FireplaceQu, dtype: int64
    ##################################################
    Attchd     813
    Detchd     360
    BuiltIn     77
    Basment     16
    CarPort      6
    2Types       1
    Name: GarageType, dtype: int64
    ##################################################
    Unf    552
    RFn    401
    Fin    320
    Name: GarageFinish, dtype: int64
    ##################################################
    TA    1216
    Fa      44
    Gd       9
    Po       2
    Ex       2
    Name: GarageQual, dtype: int64
    ##################################################
    TA    1225
    Fa      33
    Gd       7
    Po       6
    Ex       2
    Name: GarageCond, dtype: int64
    ##################################################
    Y    1237
    N      71
    P      30
    Name: PavedDrive, dtype: int64
    ##################################################
    Fa    1
    Gd    1
    Name: PoolQC, dtype: int64
    ##################################################
    MnPrv    143
    GdPrv     52
    GdWo      51
    MnWw       9
    Name: Fence, dtype: int64
    ##################################################
    Shed    36
    Othr     1
    Gar2     1
    Name: MiscFeature, dtype: int64
    ##################################################
    WD       1162
    New       111
    COD        39
    ConLD       7
    ConLI       5
    ConLw       5
    CWD         4
    Oth         3
    Con         2
    Name: SaleType, dtype: int64
    ##################################################
    Normal     1106
    Partial     113
    Abnorml      87
    Family       19
    Alloca        9
    AdjLand       4
    Name: SaleCondition, dtype: int64
    ##################################################
    


```python
li_cat_feats = list(cate) 
nr_rows = 15 
nr_cols = 3 
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3)) 
for r in range(0,nr_rows): 
    for c in range(0,nr_cols): 
        i = r*nr_cols+c 
        if i < len(li_cat_feats): 
            sns.boxplot(x=li_cat_feats[i], y=train["SalePrice"], data=train, ax = axs[r][c]) 
plt.tight_layout() 
plt.show()


```


![output_38_0](https://user-images.githubusercontent.com/45659433/119081393-d76a4b80-ba36-11eb-88a6-da3749957e66.png)



```python
num_strong_corr = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars', 'FullBath','YearBuilt','YearRemodAdd'] 
num_weak_corr = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
                 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'] 
catg_strong_corr = ['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 
                    'Electrical', 'KitchenQual', 'SaleType'] 
catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 
                  'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation',
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'Functional', 'FireplaceQu',
                  'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                  'SaleCondition' ]


```

## 3. feature engineering

### 1) 왜도 첨도 해결


```python
f, ax = plt.subplots(1, 1, figsize = (10,6)) 
g = sns.distplot(train["SalePrice"], color = "b", label="Skewness: {:2f}".format(train["SalePrice"].skew()), ax=ax) 
g = g.legend(loc = "best") 

print("Skewness: %f" % train["SalePrice"].skew()) 
print("Kurtosis: %f" % train["SalePrice"].kurt())
```

    Skewness: 1.287364
    Kurtosis: 2.555029
    

![output_42_1](https://user-images.githubusercontent.com/45659433/119081394-d802e200-ba36-11eb-9cf8-3e68d46934d6.png)


- 비대칭도와 첨도를 해결하기 위해 데이터 분포에 log


```python
train["SalePrice_Log"] = train["SalePrice"].map(lambda i:np.log(i) if i>0 else 0) 

f, ax = plt.subplots(1, 1, figsize = (10,6)) 
g = sns.distplot(train["SalePrice_Log"], color = "b", label="Skewness: {:2f}".format(train["SalePrice_Log"].skew()), ax=ax) 
g = g.legend(loc = "best") 

print("Skewness: %f" % train['SalePrice_Log'].skew()) 
print("Kurtosis: %f" % train['SalePrice_Log'].kurt()) 
train.drop('SalePrice', axis= 1, inplace=True)


```

    Skewness: -0.032026
    Kurtosis: 0.571897
    


![output_44_1](https://user-images.githubusercontent.com/45659433/119081395-d802e200-ba36-11eb-9b86-c86307ac1a32.png)


### 2) 결측치 해결

- 결측의 의미가 있다 없다 중 하나일 경우


```python
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu', 'GarageQual','GarageCond','GarageFinish',
               'GarageType', 'Electrical', 'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st', 
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2', 'MSZoning', 'Utilities'] 
for col in cols_fillna: 
    train[col].fillna('None',inplace=True) 
    test[col].fillna('None',inplace=True)

```


```python
total = train.isnull().sum().sort_values(ascending=False) 
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False) 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
missing_data.head(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LotFrontage</th>
      <td>229</td>
      <td>0.171151</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>65</td>
      <td>0.048580</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>7</td>
      <td>0.005232</td>
    </tr>
    <tr>
      <th>Id</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.fillna(train.mean(), inplace=True) 
test.fillna(test.mean(), inplace=True)

```


```python
train.isnull().sum().sum(), test.isnull().sum().sum()
```




    (0, 0)




```python
id_test = test['Id'] 
to_drop_num = num_weak_corr 
to_drop_catg = catg_weak_corr 
cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 
for df in [train, test]: 
    df.drop(cols_to_drop, inplace= True, axis = 1)


```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSZoning</th>
      <th>Neighborhood</th>
      <th>Condition2</th>
      <th>OverallQual</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrType</th>
      <th>ExterQual</th>
      <th>BsmtQual</th>
      <th>TotalBsmtSF</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>KitchenQual</th>
      <th>GarageCars</th>
      <th>SaleType</th>
      <th>SalePrice_Log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RL</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>7</td>
      <td>2003</td>
      <td>2003</td>
      <td>BrkFace</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>856</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1710</td>
      <td>2</td>
      <td>Gd</td>
      <td>2</td>
      <td>WD</td>
      <td>12.247694</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RL</td>
      <td>Veenker</td>
      <td>Norm</td>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>None</td>
      <td>TA</td>
      <td>Gd</td>
      <td>1262</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>2</td>
      <td>TA</td>
      <td>2</td>
      <td>WD</td>
      <td>12.109011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RL</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>7</td>
      <td>2001</td>
      <td>2002</td>
      <td>BrkFace</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>920</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1786</td>
      <td>2</td>
      <td>Gd</td>
      <td>2</td>
      <td>WD</td>
      <td>12.317167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RL</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>7</td>
      <td>1915</td>
      <td>1970</td>
      <td>None</td>
      <td>TA</td>
      <td>TA</td>
      <td>756</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1717</td>
      <td>1</td>
      <td>Gd</td>
      <td>3</td>
      <td>WD</td>
      <td>11.849398</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RL</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>8</td>
      <td>2000</td>
      <td>2000</td>
      <td>BrkFace</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>1145</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>2198</td>
      <td>2</td>
      <td>Gd</td>
      <td>3</td>
      <td>WD</td>
      <td>12.429216</td>
    </tr>
  </tbody>
</table>
</div>



- Categorical Data들을 수치형으로 변환


```python
catg_list = catg_strong_corr.copy() 
catg_list.remove('Neighborhood')  #너무 많아서
for catg in catg_list :
    sns.violinplot(x=catg, y=train["SalePrice_Log"], data=train) 
    plt.show()


```

![output_54_0](https://user-images.githubusercontent.com/45659433/119081396-d89b7880-ba36-11eb-8d4a-4bad73619968.png)



![output_54_1](https://user-images.githubusercontent.com/45659433/119081397-d89b7880-ba36-11eb-94b1-78e4b9fb7572.png)



![output_54_2](https://user-images.githubusercontent.com/45659433/119081398-d9340f00-ba36-11eb-9437-ef497543dbf6.png)



![output_54_3](https://user-images.githubusercontent.com/45659433/119081400-d9340f00-ba36-11eb-800f-62fbb97a3d01.png)



![output_54_4](https://user-images.githubusercontent.com/45659433/119081401-d9cca580-ba36-11eb-958c-7ce85d2ebb35.png)



![output_54_5](https://user-images.githubusercontent.com/45659433/119081402-d9cca580-ba36-11eb-86d6-fa271aa21259.png)



![output_54_6](https://user-images.githubusercontent.com/45659433/119081404-da653c00-ba36-11eb-8ae4-4e5dfb12f12c.png)


![output_54_7](https://user-images.githubusercontent.com/45659433/119081407-da653c00-ba36-11eb-93ce-f559064ac82b.png)



![output_54_8](https://user-images.githubusercontent.com/45659433/119081408-dafdd280-ba36-11eb-9f1a-68b2bc8ce5c7.png)



```python
fig, ax = plt.subplots() 
fig.set_size_inches(16, 5) 
sns.violinplot(x='Neighborhood', y=train["SalePrice_Log"], data=train, ax=ax) 
plt.xticks(rotation=45) 
plt.show()

```


![output_55_0](https://user-images.githubusercontent.com/45659433/119081410-dafdd280-ba36-11eb-8e0f-80567f3ec3e8.png)


- 각 범주의 SalePrice_Log 평균


```python
for catg in catg_list :
    g = train.groupby(catg)["SalePrice_Log"].mean() 
    print(g)


```

    MSZoning
    C (all)    10.960733
    FV         12.246616
    RH         11.700602
    RL         12.069474
    RM         11.677434
    Name: SalePrice_Log, dtype: float64
    Condition2
    Artery    11.570036
    Feedr     11.734675
    Norm      12.012766
    PosN      12.860999
    RRAn      11.827043
    RRNn      11.435329
    Name: SalePrice_Log, dtype: float64
    MasVnrType
    BrkCmn     11.757397
    BrkFace    12.150038
    None       11.884715
    Stone      12.391766
    Name: SalePrice_Log, dtype: float64
    ExterQual
    Ex    12.727937
    Fa    11.274881
    Gd    12.295102
    TA    11.828332
    Name: SalePrice_Log, dtype: float64
    BsmtQual
    Ex      12.595497
    Fa      11.585725
    Gd      12.170320
    None    11.572903
    TA      11.791989
    Name: SalePrice_Log, dtype: float64
    CentralAir
    N    11.471479
    Y    12.042717
    Name: SalePrice_Log, dtype: float64
    Electrical
    FuseA    11.641350
    FuseF    11.565700
    FuseP    11.256345
    Mix      11.112448
    None     12.028739
    SBrkr    12.047595
    Name: SalePrice_Log, dtype: float64
    KitchenQual
    Ex    12.582467
    Fa    11.514567
    Gd    12.209558
    TA    11.800968
    Name: SalePrice_Log, dtype: float64
    SaleType
    COD      11.788173
    CWD      12.198344
    Con      12.483911
    ConLD    11.900627
    ConLI    12.044878
    ConLw    11.769706
    New      12.427026
    Oth      11.675295
    WD       11.979355
    Name: SalePrice_Log, dtype: float64
    


```python
# 'MSZoning' 
msz_catg2 = ['RM', 'RH'] 
msz_catg3 = ['RL', 'FV'] 
# Neighborhood 
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker'] 
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr'] 
# Condition2 
cond2_catg2 = ['Norm', 'RRAe'] 
cond2_catg3 = ['PosA', 'PosN'] 
# SaleType 
SlTy_catg1 = ['Oth'] 
SlTy_catg3 = ['CWD'] 
SlTy_catg4 = ['New', 'Con']


```

- 범주를 수치형으로 변환


```python
for df in [train, test]: 
    df['MSZ_num'] = 1 
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2 
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3 
    df['NbHd_num'] = 1 
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2 
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3 
    df['Cond2_num'] = 1 
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2 
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3 
    df['Mas_num'] = 1 
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    df['ExtQ_num'] = 1 
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2 
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3 
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4 
    df['BsQ_num'] = 1 
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2 
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3 
    df['CA_num'] = 0 
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1 
    df['Elc_num'] = 1 
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 
    df['KiQ_num'] = 1 
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2 
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3 
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4 
    df['SlTy_num'] = 2 
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1 
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3 
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4


```


```python
new_col_HM = train[['SalePrice_Log', 'MSZ_num', 'NbHd_num', 'Cond2_num', 'Mas_num', 'ExtQ_num', 'BsQ_num', 'CA_num',
                    'Elc_num', 'KiQ_num', 'SlTy_num']] 
colormap = plt.cm.PuBu 
plt.figure(figsize=(10, 8)) 
plt.title("Correlation of New Features", y = 1.05, size = 15) 
sns.heatmap(new_col_HM.corr(), linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = "white",
            annot = True, annot_kws = {"size" : 12})


```




    <matplotlib.axes._subplots.AxesSubplot at 0x202acf910a0>




![output_61_1](https://user-images.githubusercontent.com/45659433/119081412-db966900-ba36-11eb-84fb-9a811cdb522f.png)


- 의미없는 범주형 변수 삭제 (NbHd_num, ExtQ_num, BsQ_num, KiQ_num 제외)


```python
train.drop(['MSZoning','Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir',
               'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 
               'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True) 

test.drop(['MSZoning', 'Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual',
              'CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 
              'Elc_num', 'SlTy_num'], axis = 1, inplace = True)


```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OverallQual</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>TotalBsmtSF</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>GarageCars</th>
      <th>SalePrice_Log</th>
      <th>MSZ_num</th>
      <th>NbHd_num</th>
      <th>ExtQ_num</th>
      <th>BsQ_num</th>
      <th>KiQ_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>2003</td>
      <td>2003</td>
      <td>856</td>
      <td>1710</td>
      <td>2</td>
      <td>2</td>
      <td>12.247694</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>1262</td>
      <td>1262</td>
      <td>2</td>
      <td>2</td>
      <td>12.109011</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>2001</td>
      <td>2002</td>
      <td>920</td>
      <td>1786</td>
      <td>2</td>
      <td>2</td>
      <td>12.317167</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>1915</td>
      <td>1970</td>
      <td>756</td>
      <td>1717</td>
      <td>1</td>
      <td>3</td>
      <td>11.849398</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>2000</td>
      <td>2000</td>
      <td>1145</td>
      <td>2198</td>
      <td>2</td>
      <td>3</td>
      <td>12.429216</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OverallQual</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>TotalBsmtSF</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
      <th>GarageCars</th>
      <th>MSZ_num</th>
      <th>NbHd_num</th>
      <th>ExtQ_num</th>
      <th>BsQ_num</th>
      <th>KiQ_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>1961</td>
      <td>1961</td>
      <td>882.0</td>
      <td>896</td>
      <td>1</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1958</td>
      <td>1958</td>
      <td>1329.0</td>
      <td>1329</td>
      <td>1</td>
      <td>1.0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>1997</td>
      <td>1998</td>
      <td>928.0</td>
      <td>1629</td>
      <td>2</td>
      <td>2.0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>1998</td>
      <td>1998</td>
      <td>926.0</td>
      <td>1604</td>
      <td>2</td>
      <td>2.0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>1992</td>
      <td>1992</td>
      <td>1280.0</td>
      <td>1280</td>
      <td>2</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## 4. 모델


```python
from sklearn.model_selection import train_test_split 
from sklearn import metrics 

X_train = train.drop("SalePrice_Log", axis = 1).values 
target_label = train["SalePrice_Log"].values 
X_test = test.values 
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size = 0.3, random_state = 2000) #train을 70% 
```

### 1) XGBoost


```python
import xgboost 

regressor = xgboost.XGBRegressor(colsample_bytree = 0.4603, learning_rate = 0.06, min_child_weight = 1.8, max_depth= 4, subsample = 0.52, n_estimators = 2000, random_state= 7, ntrhead = -1) 
regressor.fit(X_tr,y_tr)

```

    [17:54:08] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    




    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=0.4603, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.06, max_delta_step=0, max_depth=4,
                 min_child_weight=1.8, missing=nan, monotone_constraints='()',
                 n_estimators=2000, n_jobs=6, ntrhead=-1, num_parallel_tree=1,
                 random_state=7, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 subsample=0.52, tree_method='exact', validate_parameters=1,
                 verbosity=None)




```python
y_hat = regressor.predict(X_tr) 
plt.scatter(y_tr, y_hat, alpha = 0.2) 
plt.xlabel('Targets (y_tr)',size=18) 
plt.ylabel('Predictions (y_hat)',size=18) 
plt.show()


```

![output_70_0](https://user-images.githubusercontent.com/45659433/119081413-dc2eff80-ba36-11eb-985e-6e86284810a2.png)



```python
regressor.score(X_tr,y_tr)
```




    0.9942432559890882



- validation 데이터


```python
y_hat_test = regressor.predict(X_vld) 
plt.scatter(y_vld, y_hat_test, alpha=0.2) 
plt.xlabel('Targets (y_vld)',size=18) 
plt.ylabel('Predictions (y_hat_test)',size=18) 
plt.show()


```

![output_73_0](https://user-images.githubusercontent.com/45659433/119081414-dc2eff80-ba36-11eb-9f87-d3a3bfec0c53.png)



```python
regressor.score(X_vld,y_vld)
```




    0.8415909007883536




```python
from sklearn.model_selection import cross_val_score 

accuracies = cross_val_score(estimator = regressor, X = X_tr, y = y_tr, cv = 10)

```

    [17:54:09] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:54:10] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:54:11] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:54:11] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:54:12] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:54:13] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:54:14] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:54:15] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:54:16] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    [17:54:16] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:573: 
    Parameters: { "ntrhead" } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    
    


```python
print(accuracies.mean()) 
print(accuracies.std())

```

    0.8116812095589777
    0.06551870893118561
    

## 제출


```python
use_logvals = 1 
pred_xgb = regressor.predict(X_test) 
sub_xgb = pd.DataFrame() 
sub_xgb['Id'] = id_test 
sub_xgb['SalePrice'] = pred_xgb 
if use_logvals == 1: 
    sub_xgb['SalePrice'] = np.exp(sub_xgb['SalePrice']) 
    sub_xgb.to_csv('houseprice_1.csv',index=False)


```

## 2) DecisionTreeRegressor


```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
 
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


```


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
 
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_tr, y_tr)
melb_preds = forest_model.predict(X_vld)
print(mean_absolute_error(y_vld, melb_preds))

```

    0.09997253124191095
    


```python
preds = forest_model.predict(X_test)
```


```python
accuracies = cross_val_score(estimator = forest_model, X = X_tr, y = y_tr, cv = 10)
print(accuracies.mean()) 
print(accuracies.std())

```

    0.8400687051104556
    0.05883212599174881
    


```python
sub_xgb = pd.DataFrame() 
sub_xgb['Id'] = id_test 
sub_xgb['SalePrice'] = np.exp(preds) 
sub_xgb.to_csv('houseprice_2.csv',index=False)

```

참고: https://hong-yp-ml-records.tistory.com/3?category=818779

DecisionTreeRegressor이 더 좋게 나왔답니다.


```python

```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTc2MjQ0NTM4NV19
-->
