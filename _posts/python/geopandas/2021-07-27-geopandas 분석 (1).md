## geopandas 분석방법 (1)

#### #위치가 포함된 데이터를 이용하여 데이터 분석하기!
#### #데이터 로드부터 저장까지

**< 순서 >**  
1) geojson 데이터 읽는 방법  
2) geojson을 데이터 프레임 형식으로 만들기  
3) geojson 저장방법  




```python
import geopandas as gpd
from geopandas import GeoDataFrame

import pandas as pd
import json
```

### 1) geojson 읽는 방법


```python
state_geo = 'shp_to_gep.geojson'
cen_str=json.load(open(state_geo, encoding='utf-8'))
cen_str
```




    {'type': 'FeatureCollection',
     'crs': {'type': 'name',
      'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
     'features': [{'type': 'Feature',
       'properties': {'TOT_REG_CD': '1101053010006',
        'ADM_NM': '사직동',
        'ADM_CD': '1101053'},
       'geometry': {'type': 'Polygon',
        'coordinates': [[[126.97033304133119, 37.57911880232895],
          [126.97039244976332, 37.579099248302775],
          [126.97039708040217, 37.57909752768807],
          [126.97043980256053, 37.57907503249067],
          [126.97044812634958, 37.5790709182569],
          [126.97045250813798, 37.57906883776548],
          [126.97047415959725, 37.57904888720932],
          [126.9704975473185, 37.57902780149314]]]}},
      ...]}



### 2) geojson을 데이터 프레임 형식으로 만들기
- 1번째 방법


```python
cen_str_df=gpd.read_file('D:\모빌리티\유동인구 데이터\shp_to_gep.geojson',driver='GeoJSON')
cen_str_df=cen_str_df[['ADM_NM','ADM_CD','TOT_REG_CD','geometry']]
cen_str_df
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
      <th>ADM_NM</th>
      <th>ADM_CD</th>
      <th>TOT_REG_CD</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>사직동</td>
      <td>1101053</td>
      <td>1101053010006</td>
      <td>POLYGON ((126.97033 37.57912, 126.97039 37.579...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>사직동</td>
      <td>1101053</td>
      <td>1101053010001</td>
      <td>POLYGON ((126.96613 37.57496, 126.96632 37.574...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>사직동</td>
      <td>1101053</td>
      <td>1101053010003</td>
      <td>POLYGON ((126.96645 37.57883, 126.96647 37.578...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>사직동</td>
      <td>1101053</td>
      <td>1101053010002</td>
      <td>POLYGON ((126.96877 37.57823, 126.96878 37.578...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>사직동</td>
      <td>1101053</td>
      <td>1101053010005</td>
      <td>POLYGON ((126.97399 37.57823, 126.97400 37.578...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19148</th>
      <td>동화동</td>
      <td>1102073</td>
      <td>1102073010501</td>
      <td>POLYGON ((127.01764 37.56250, 127.01763 37.562...</td>
    </tr>
    <tr>
      <th>19149</th>
      <td>명동</td>
      <td>1102055</td>
      <td>1102055030001</td>
      <td>POLYGON ((126.98893 37.56810, 126.98895 37.568...</td>
    </tr>
    <tr>
      <th>19150</th>
      <td>장충동</td>
      <td>1102058</td>
      <td>1102058040004</td>
      <td>POLYGON ((127.00338 37.55903, 127.00344 37.559...</td>
    </tr>
    <tr>
      <th>19151</th>
      <td>보라매동</td>
      <td>1121052</td>
      <td>1121052010002</td>
      <td>POLYGON ((126.92720 37.49429, 126.92709 37.493...</td>
    </tr>
    <tr>
      <th>19152</th>
      <td>보라매동</td>
      <td>1121052</td>
      <td>1121052010001</td>
      <td>POLYGON ((126.92591 37.49144, 126.92596 37.491...</td>
    </tr>
  </tbody>
</table>
<p>19153 rows × 4 columns</p>
</div>



- 2번째 방법


```python
geo=json.load(open('shp_to_gep.geojson',encoding='utf-8'))
f = pd.json_normalize(geo['features'])
f
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
      <th>type</th>
      <th>properties.TOT_REG_CD</th>
      <th>properties.ADM_NM</th>
      <th>properties.ADM_CD</th>
      <th>geometry.type</th>
      <th>geometry.coordinates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Feature</td>
      <td>1101053010006</td>
      <td>사직동</td>
      <td>1101053</td>
      <td>Polygon</td>
      <td>[[[126.97033304133119, 37.57911880232895], [12...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Feature</td>
      <td>1101053010001</td>
      <td>사직동</td>
      <td>1101053</td>
      <td>Polygon</td>
      <td>[[[126.96613384750641, 37.574957380849554], [1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Feature</td>
      <td>1101053010003</td>
      <td>사직동</td>
      <td>1101053</td>
      <td>Polygon</td>
      <td>[[[126.96645292161898, 37.5788253347882], [126...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Feature</td>
      <td>1101053010002</td>
      <td>사직동</td>
      <td>1101053</td>
      <td>Polygon</td>
      <td>[[[126.96877457487757, 37.57822935210929], [12...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Feature</td>
      <td>1101053010005</td>
      <td>사직동</td>
      <td>1101053</td>
      <td>Polygon</td>
      <td>[[[126.9739856241019, 37.57823267082909], [126...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19148</th>
      <td>Feature</td>
      <td>1102073010501</td>
      <td>동화동</td>
      <td>1102073</td>
      <td>Polygon</td>
      <td>[[[127.01763808390233, 37.562499122782185], [1...</td>
    </tr>
    <tr>
      <th>19149</th>
      <td>Feature</td>
      <td>1102055030001</td>
      <td>명동</td>
      <td>1102055</td>
      <td>Polygon</td>
      <td>[[[126.98892726910591, 37.568102798887075], [1...</td>
    </tr>
    <tr>
      <th>19150</th>
      <td>Feature</td>
      <td>1102058040004</td>
      <td>장충동</td>
      <td>1102058</td>
      <td>Polygon</td>
      <td>[[[127.00338472431302, 37.559025029043795], [1...</td>
    </tr>
    <tr>
      <th>19151</th>
      <td>Feature</td>
      <td>1121052010002</td>
      <td>보라매동</td>
      <td>1121052</td>
      <td>Polygon</td>
      <td>[[[126.92719682242266, 37.494286392496186], [1...</td>
    </tr>
    <tr>
      <th>19152</th>
      <td>Feature</td>
      <td>1121052010001</td>
      <td>보라매동</td>
      <td>1121052</td>
      <td>Polygon</td>
      <td>[[[126.92590640761341, 37.49143781173751], [12...</td>
    </tr>
  </tbody>
</table>
<p>19153 rows × 6 columns</p>
</div>



### 3) geojson 저장방법


```python
cen_str_df.to_file('cen_str_df.geojson', driver='GeoJSON')
```
