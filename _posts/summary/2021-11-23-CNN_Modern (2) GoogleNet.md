---

author: Hone Ye ji
categories: 
 - deep learning
 - CNN 
tags: 
 - deep learning

toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---
## GoogleNet
- 22 layers
- inception 모듈 : 1x1 conv 사용 (채널수 줄여 계산비용을 줄이고 복잡도 줄이기위해)
- 다층 계층을 직렬로 연결
- 초반(AlexNet과 LeNet과 유사)
- Global Avg Pool (FC layer 같은 역할)
-

## Batch normalization
### 1) 장점
- 학습속도 개선 (빠른 속도로 가능)
- 가중치 초기화에 대한 민감도 감소
- 모델의 일반화(regularization)효과

### 2) mini-batch : 데이터를 여러개로 쪼개는 것을 의미
### 3) 정규화
#### 3-1. 정규화 1 :  min-max
$\hat x =$ $\frac{ x-x_{min}} { x_{max} - x_{min}}$
- 입력데이터의 학습속도 개선(변수 scale영향 없앰)
#### 3-2. 정규화 2: z-score(normalization)
$\hat x =$ $\frac{ x-E[X]} { Var[X]}$
- 분포가 0을 중심으로 스케일 됨

### 4) TRAIN 

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk4NDk3NjUzMywtMTA3NDE3Njk4MiwtOT
A3MzA3MTgxXX0=
-->