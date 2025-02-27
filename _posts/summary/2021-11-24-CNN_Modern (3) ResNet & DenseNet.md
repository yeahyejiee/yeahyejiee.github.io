---

author: Hone Ye ji
categories: 
 - deep learning
 - CNN
tags: 
 - deep learning
use_math: true
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
### (1) ResNet과 비교

- 이전 layer의 feature map을 다음 layer의 입력과 연결하는 방식
- ResNet : feature map끼리 더하기를 해주는 방식 (정보들을 온전히 흘러가는 것을 방해)
- DenseNet: feature map끼리 concatenation(연속)을 해주는 방식

![DenseNet — Organize everything I know documentation](https://oi.readthedocs.io/en/latest/_images/cnn_vs_resnet_vs_densenet.png)

### (2) convolution block
![../_images/densenet-block.svg](https://d2l.ai/_images/densenet-block.svg)

### (3) 특징
- VGG와 같은 3X3 filter
- 밀도블록(dense block): convolution block 구조, 각 conv block의 입력과 출력 연결
- Transition layer: 모델의 복잡성 제어 (1x1 conv : 채널수 줄이기, average pooling 이용: 높이, 너비를 반으로)
- 4개의 조밀한 Block
- 출력: 입력채널 + conv block 개수 x 출력채널(growth rate)
	- growth rate: conv block의 출력 채널 수는 출력의 증가를 제어

### (4) transition block
batch normalization  ->  conv 1x1  -> Avg pool 2x2 ,stride 2


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1NTA0NzYzNTYsLTEwOTc0OTY2OTksLT
E1ODM2MDU2ODBdfQ==
-->
