# 1. Single Hidden Layer


single hidden layer란 말그대로 은닉층이 한개의 층으로 구성이 되어있다는 것이다.  수식에 따르면 밑과 같다.
$$input:    x \\h= \sigma (w_{1}x +b_{1}) \\  o=w_{2}^{T}+b_{2}$$

## 1) Activation Function (활성화함수)
![활성화 함수와 옵티마이저 공부 | kite_mo](https://wikidocs.net/images/page/60683/simple-neural-network.png)

Activation Function (활성화함수)는 입력된 데이터의 가중 합을 출력 신호로 변환하는 함수이다. 종류에는 Sigmoid, Tanh, ReLU 등이 있으며 소개할 함수이다.

![ML — Sigmoid 대신 ReLU? 상황에 맞는 활성화 함수 사용하기 | by Minkyeong Kim | Medium](https://miro.medium.com/max/666/1*nrxtwp6rzqdFhgYh0x-eVw.png)


### (1) Sigmoid
0에서 1까지의 범위를 가진다. 밑 그림처럼 미분할경우 최고점이 0에서 0.25이며 0에서 멀어질수록 0에 수렴하는 것을 볼 수 있다. 이에 따라 딥러닝 사용할때 weight가 소멸되는 문제점이 나타난다.

$$sigmoid(x)=\frac{1}{1+e^{−x}}$$
$$\frac{d}{dx} sigmoid(x) = \frac{e^{-x} }{(1+e^{-x})^{2}} = sigmoid(x)(1-sigmoid(x)) $$


![딥 러닝에서 알아야 할 7 가지 인기있는 활성화 함수와 Keras 및 TensorFlow 2와 함께 사용하는 방법](https://ichi.pro/assets/images/max/724/1*mOyWsQ0HuPYLZ0B8c4rH-A.png)

### (2) Tanh
출력 범위는 -1에서 1사이다.  밑 그림 왼쪽은  tanh의 그래프이며 오른쪽은 미분했을때 나오는 그래프이다. 왼쪽일 경우 -1~1사이로 나오며 오른쪽은 0에서 1의 값을 가지며 0에서 멀어질 수록 0, 즉 소멸된다.

$$tanh(x)=\frac{1-e^{-2x}}{1+e^{−2x}}$$
$$\frac{d}{dx} tanh(x) = 1-tanh^2(x) $$

![딥 러닝에서 알아야 할 7 가지 인기있는 활성화 함수와 Keras 및 TensorFlow 2와 함께 사용하는 방법](https://ichi.pro/assets/images/max/724/1*jW-JYhK4I-CbahDaapWzXg.png)

### (3) ReLU
ReLU는 미분 값이 사라지지않아서 많이 사용된다. 음수를 가지지않고 0이상의 값을 가진다. 즉, 양수다. 미분할경우, 음수는 0을 양수는 1의 값을 가진다.
$$tanh(x)=max(x,0) $$

![딥러닝 - 활성화 함수(Activation) 종류 및 비교 : 네이버 블로그](https://mblogthumb-phinf.pstatic.net/MjAyMDAyMjVfOTIg/MDAxNTgyNjA4MzI2NDA5.e0VyX0yrhE5gtfPjni7IxF5kpArCeByreQsdOMB0240g.CWwTi57bPtAK6C7eLmRn1ED2RE8Lm_C6sVIwMGJS1Akg.PNG.handuelly/image.png?type=w800)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4OTQzNTM0NTMsNzQ4NzYzODNdfQ==
-->