# 1. Single Hidden Layer


single hidden layer란 말그대로 은닉층이 한개의 층으로 구성이 되어있다는 것이다.  수식에 따르면 밑과 같다.
$$input:    x \\h= \sigma (w_{1}x +b_{1}) \\  o=w_{2}^{T}+b_{2}$$

## 1) Activation Function (활성화함수)
![퀀티랩 블로그 - 딥러닝 발전 과정](https://lh3.googleusercontent.com/proxy/p3_KJV_XOazdJmbz4AOLmG2ny7Xbj14AmoMZ1VOmpg1Bbc3mIpXsxX4qMWljqvJn26tl7kRDfgML-XEUnO2bFayXToqD6gVxTMyf1AYcjnP6BzzOF9Yt)

Activation Function (활성화함수)는 입력된 데이터의 가중 합을 출력 신호로 변환하는 함수이다. 종류에는 Sigmoid, Tanh, ReLU 등이 있으며 소개할 함수이다.

![ML — Sigmoid 대신 ReLU? 상황에 맞는 활성화 함수 사용하기 | by Minkyeong Kim | Medium](https://miro.medium.com/max/666/1*nrxtwp6rzqdFhgYh0x-eVw.png)


### (1) Sigmoid
0에서 1까지의 범위를 가진다. 밑 그림처럼 미분할경우 최고점이 0에서 0.25이며 0에서 멀어질수록 0에 수렴하는 것을 볼 수 있다. 이에 따라 딥러닝 사용할때 weight가 소멸되는 문제점이 나타난다.

$$sigmoid(x)=\frac{1}{1+e^{−x}}$$
$$\frac{d}{dx} sigmoid(x) = \frac{e^{-x} }{(1+e^{-x})^{2}}=$$


![딥 러닝에서 알아야 할 7 가지 인기있는 활성화 함수와 Keras 및 TensorFlow 2와 함께 사용하는 방법](https://ichi.pro/assets/images/max/724/1*mOyWsQ0HuPYLZ0B8c4rH-A.png)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTAxMzc2OTgzNF19
-->