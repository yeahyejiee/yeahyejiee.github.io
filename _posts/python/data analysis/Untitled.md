# 맛집 평점, 링크, 주소 크롤링


```python
import time
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import requests
import csv
import urllib.request
import urllib.parse
import re
from bs4 import BeautifulSoup
```

### 1. 크롤링 함수 정의


```python
def storeNamePrint():

    time.sleep(0.2)
    html =driver.page_source
    soup=BeautifulSoup(html,'html.parser')

    cafe_lists=soup.select('.placelist > .PlaceItem')
    count=1
    for cafe in cafe_lists:

        temp=[]
        cafe_name=cafe.select('.head_item > .tit_name > .link_name')[0].text
        food_score=cafe.select('.rating >.score >.num')[0].text
        review=cafe.select('.rating > .review')[0].text
        link=cafe.select('.contact >.moreview')[0]['href']
        addr=cafe.select('.addr')[0].text

        #리뷰 문자열 제거 후 숫자만 반환
        review = review[3:len(review)]
        review=int(re.sub(",","",review))

        print(cafe_name, food_score, review, link, addr)
        temp.append(cafe_name)
        temp.append(food_score)  
        temp.append(review)
        temp.append(link)
        temp.append(addr)

        list.append(temp)

    f=open(filename +'.csv',"w", encoding='utf-8-sig',newline="")
    writercsv=csv.writer(f)
    header=['Name','Score','Reivew','Link','Addr']
    writercsv.writerow(header)

    for i in list:
        writercsv.writerow(i)

```

### 2. 크롬 위치 및 지도 이동 :반복을 위해서 원하는 검색어 지정


```python
list=[]

url="https://map.kakao.com/"
options=webdriver.ChromeOptions() 
options.add_argument('lang=ko_KR') #한국어

chromedriver_path='chromedriver.exe'  #크롬 드라이버 위치

driver=webdriver.Chrome(os.path.join(os.getcwd(), chromedriver_path),options=options)

# 카카오지도로 이동
driver.get(url)

# 원하는 검색어
gu_list=["강남구","강동구","강북구","강서구","관악구","광진구","구로구","금천구",
        "노원구","도봉구","동대문구","동작구","마포구","서대문구","서초구","성동구",
        "성북구","송파구","양천구","영등포구","용산구","은평구","종로구","중구","중랑구"]

searchloc=[]        
for i in range(len(gu_list)):
    searchloc.append(f"서울 {gu_list[i]} 맛집")
    
    
searchloc
```




    ['서울 강남구 맛집',
     '서울 강동구 맛집',
     '서울 강북구 맛집',
     '서울 강서구 맛집',
     '서울 관악구 맛집',
     '서울 광진구 맛집',
     '서울 구로구 맛집',
     '서울 금천구 맛집',
     '서울 노원구 맛집',
     '서울 도봉구 맛집',
     '서울 동대문구 맛집',
     '서울 동작구 맛집',
     '서울 마포구 맛집',
     '서울 서대문구 맛집',
     '서울 서초구 맛집',
     '서울 성동구 맛집',
     '서울 성북구 맛집',
     '서울 송파구 맛집',
     '서울 양천구 맛집',
     '서울 영등포구 맛집',
     '서울 용산구 맛집',
     '서울 은평구 맛집',
     '서울 종로구 맛집',
     '서울 중구 맛집',
     '서울 중랑구 맛집']



### 3.페이지 넘기며 크롤링 하기


```python
for loc in range(len(searchloc)):
    driver=webdriver.Chrome(os.path.join(os.getcwd(), chromedriver_path),options=options)
    driver.get(url)
    filename=searchloc[loc]
    answer_loc=searchloc[loc]
    print(answer_loc)
    
    #서울시 음식점
    search_area=driver.find_element_by_xpath('//*[@id="search.keyword.query"]')
    search_area.send_keys(answer_loc) #검색
    driver.find_element_by_xpath('//*[@id="search.keyword.submit"]').send_keys(Keys.ENTER) #Enter누름
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="info.main.options"]/li[2]/a').send_keys(Keys.ENTER)  #장소버튼누름
    time.sleep(0.5)
    
    # 페이지 총 34까지
    page = 1
    for page_1 in range(0,34):

        page_1 += 1
        print("**",page_1,"**")

        driver.find_element_by_xpath(f'//*[@id="info.search.page.no{page}"]').send_keys(Keys.ENTER)

        storeNamePrint()
        
        if (page)%5==0:
            element= driver.find_element_by_xpath('//*[@id="info.search.page.next"]')
            driver.execute_script("arguments[0].click();", element)
            page=0

        page += 1
        

    driver.close()
    print(answer_loc,"크롤링완료***")

```

- 참고 : https://nostalgiaa.tistory.com/36
