---

author: Hone Ye ji
categories: 
 - geopandas
tags: 
 - geopandas
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---

## 지하철 주소를 위도와 경도로 변경하기
### 카카오 key발급 후 사용할 수 있음


```python
import requests
from urllib.parse import urlparse
import geopandas
import pandas as pd
add= pd.read_csv("서울교통공사 지하철역 주소 및 전화번호 정보.csv",engine='python',encoding='CP949')
```


```python
def address_to_loaction(address):
    global val
    url = 'https://dapi.kakao.com/v2/local/search/address.json?query='+address
    result=requests.get(urlparse(url).geturl(),
                            headers = {"Authorization": 'KakaoAK REST API 넣기'})
    json_obj=result.json()
    for document in json_obj['documents']:
        val =[document['road_address']['building_name'], document['address_name'],document['y'], document['x']]
    return val
```


```python
address_to_loaction('서울특별시 동대문구 왕산로 지하 93 (제기동) (제기동역)')
```




    ['제기동역', '서울 동대문구 왕산로 지하 93', '37.5781967284885', '127.034690599752']




```python
list=[]

for address in add["도로명주소"]:
    list.append(address_to_loaction(address))
df1=pd.DataFrame(list,columns=['1','2','lat','lon'])
df1
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
      <th>1</th>
      <th>2</th>
      <th>lat</th>
      <th>lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>청량리역</td>
      <td>서울 동대문구 왕산로 지하 205</td>
      <td>37.5802331563072</td>
      <td>127.045085886324</td>
    </tr>
    <tr>
      <th>1</th>
      <td>제기동역</td>
      <td>서울 동대문구 왕산로 지하 93</td>
      <td>37.5781967284885</td>
      <td>127.034690599752</td>
    </tr>
    <tr>
      <th>2</th>
      <td>신설동역</td>
      <td>서울 동대문구 왕산로 지하 1</td>
      <td>37.5753194830738</td>
      <td>127.024701893044</td>
    </tr>
    <tr>
      <th>3</th>
      <td>동묘앞역</td>
      <td>서울 종로구 종로 359</td>
      <td>37.5733722075697</td>
      <td>127.01675118037</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1호선 동대문역</td>
      <td>서울 종로구 종로 지하 302</td>
      <td>37.5717775660734</td>
      <td>127.011167102297</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>116</th>
      <td>4호선 이촌역</td>
      <td>서울 용산구 서빙고로 지하 83</td>
      <td>37.522520254623</td>
      <td>126.973683151228</td>
    </tr>
    <tr>
      <th>117</th>
      <td>4호선 동작역 Dongjak s.t</td>
      <td>서울 동작구 현충로 257</td>
      <td>37.5028703671011</td>
      <td>126.980268385443</td>
    </tr>
    <tr>
      <th>118</th>
      <td>총신대입구(이수)역4호선</td>
      <td>서울 동작구 동작대로 지하 117</td>
      <td>37.4875538278198</td>
      <td>126.982199707405</td>
    </tr>
    <tr>
      <th>119</th>
      <td>사당역(4호선)</td>
      <td>서울 동작구 동작대로 지하 3</td>
      <td>37.4768632653727</td>
      <td>126.981595139341</td>
    </tr>
    <tr>
      <th>120</th>
      <td>남태령역</td>
      <td>서울 서초구 과천대로 지하 816</td>
      <td>37.4648575576518</td>
      <td>126.988656544337</td>
    </tr>
  </tbody>
</table>
<p>121 rows × 4 columns</p>
</div>




```python
df1.to_csv('D:/위도경도.csv',index=False)
```


```python

```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExNTY0NjkzOTksNDI3NDI0NTM1LC0xNz
c0MDY4ODU5XX0=
-->