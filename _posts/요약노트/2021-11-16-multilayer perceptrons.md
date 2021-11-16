# 1. Single Hidden Layer


single hidden layer란 말그대로 은닉층이 한개의 층으로 구성이 되어있다는 것이다.  수식에 따르면 밑과 같다.
$$input:    x \\h= \sigma (w_{1}x +b_{1}) \\  o=w_{2}^{T}+b_{2}$$

## 1) Activation Function (활성화함수)

Activation Function (활성화함수)는 입력된 데이터의 가중 합을 출력 신호로 변환하는 함수이다. 종류에는 Sigmoid, Tanh, ReLU 등이 있으며 소개할 함수이다.

![ML — Sigmoid 대신 ReLU? 상황에 맞는 활성화 함수 사용하기 | by Minkyeong Kim | Medium](https://miro.medium.com/max/666/1*nrxtwp6rzqdFhgYh0x-eVw.png)


### (1) Sigmoid

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA4MDYyMTk2XX0=
-->