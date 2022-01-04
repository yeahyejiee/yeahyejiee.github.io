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
![../_images/residual-block.svg](https://d2l.ai/_images/residual-block.svg)
- 박스부분 : residual mapping( input $x$를 제외한 나머지 학습)
- x: residual connection(input $x$를 더하기 연산자로 전달)
- 더 빠른 전파 가능

![../_images/resnet-block.svg](https://d2l.ai/_images/resnet-block.svg)

![../_images/resnet18.svg](https://d2l.ai/_images/resnet18.svg)


### (3) 특징
- VGG의 3x3 conv layer를 따른다
- 채널수 변경 -> 1x1 conv 도입 : 입력을 원하는 모양으로 바꾸기위해
- 잔차 블록으로 구성된 4개의 모듈 (첫 모듈은 동일한 수의 출력 채널을 가짐)
- filter: 3x3 conv -> batch 정규화 -> Relu 활성화
- GlobalAvepool 사용
- $F(x)-x$(잔차 블록)가 0이 나올때까지
- GoogleNet과 유사하지만 더 단순하고 수정하기 쉬움
- 입력은 여러 계층에 걸쳐 residual connection을 통해 더 빨리 전파



## 2. DenseNet

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc1NTY1NzYwNSwtMTA5NzQ5NjY5OSwtMT
U4MzYwNTY4MF19
-->