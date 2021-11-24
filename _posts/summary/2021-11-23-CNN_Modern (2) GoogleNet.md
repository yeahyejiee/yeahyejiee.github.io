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
 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwODYzNDc1NjQsLTkwNzMwNzE4MV19
-->