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


## 1. ResNet

### (1) degradation problem
: 모델의 정확도가 어느 순간부터 정체되고 레이어가 더 깊어질수록 성능이 더 나빠짐.  ex) gradient vanishing, gradient exploding

### (2) Residual block 
### (3) 특징
- VGG의 3x3 conv layer를 따른다
- 채널수 변경 -> 1x1 conv 도입 : 입력을 원하는 모양으로 바꾸기위해
- 잔차 블록으로 구성된 4개의 모듈 (첫 모듈은 동일한 수의 출력 채널을 가짐)
- filter: 3x3 conv -> batch 정규화 -> Relu 활성화
- 
## 2. DenseNet

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTcwOTgxMjE2OCwtMTA5NzQ5NjY5OSwtMT
U4MzYwNTY4MF19
-->