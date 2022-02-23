---

author: Hone Ye ji
categories: 
 - deep learning
tags: 
 - deep learning
use_math: true
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---
딥러닝에서 hidden layer가 1개일수도 그 이상일 수도 있는데 그때 적용되는 활성화함수 종류에 대해 알아본다.

# 1. Single Hidden Layer


single hidden layer란 말그대로 은닉층이 한개의 층으로 구성이 되어있다는 것이다.  수식에 따르면 밑과 같다. 

$$input:x \\  h= \sigma (w_{1}x +b_{1}) \\  o=w_{2}^{T}h+b_{2}$$

## 1) Activation Function (활성화함수)
![활성화 함수와 옵티마이저 공부 | kite_mo](https://wikidocs.net/images/page/60683/simple-neural-network.png)

Activation Function (활성화함수)는 입력된 데이터의 가중 합을 출력 신호로 변환하는 함수이다. 종류에는 Sigmoid, Tanh, ReLU 등이 있으며 소개할 함수이다.

![image](https://user-images.githubusercontent.com/45659433/142157395-23d87ce1-cd6e-43f3-8cf3-e9ac07ed9ba9.png)


### (1) Sigmoid
파란색 선에서 보는 것과 같이 0에서 1까지의 범위를 가진다.  미분할경우, 빨간 선처럼 최고점이 0에서 0.25이며 0에서 멀어질수록 0에 수렴하는 것을 볼 수 있다. 이에 따라 딥러닝 사용할때 weight가 소멸되는 문제점이 나타난다.

$$sigmoid(x)=\frac{1}{1+e^{−x}}$$
$$\frac{d}{dx} sigmoid(x) = \frac{e^{-x} }{(1+e^{-x})^{2}} = sigmoid(x)(1-sigmoid(x)) $$


![시그모이드](https://miro.medium.com/max/2400/1*6A3A_rt4YmumHusvTvVTxw.png)

### (2) Tanh
출력 범위는 -1에서 1사이다.  밑 그림 빨간색 선은  tanh의 그래프이며 초록색 선은 미분했을때 나오는 그래프이다. 빨간색일 경우 -1~1사이로 나오며 초록색은 0에서 1의 값을 가지며 0에서 멀어질 수록 0, 즉 소멸된다.

$$tanh(x)=\frac{1-e^{-2x}}{1+e^{−2x}}$$
$$\frac{d}{dx} tanh(x) = 1-tanh^2(x) $$

![딥 러닝에서 알아야 할 7 가지 인기있는 활성화 함수와 Keras 및 TensorFlow 2와 함께 사용하는 방법](https://lh5.googleusercontent.com/S38UqpWR7-FjF5wPFWgvnaccIWMieP5lDJZFE5v2-0Sl8PlX6-5uglLxDtzzPuxHxaUEAStV0O41fgNan9Z_590hY9y71X-bEfTifVsdhJKrr2LEXLocQtiMNDFLjF6COLuKsqYh)

### (3) ReLU
ReLU는 미분 값이 사라지지않아서 많이 사용된다. 음수를 가지지않고 0이상의 값을 가진다. 즉, 양수다. 미분할경우, 음수는 0을 양수는 1의 값을 가진다.
$$tanh(x)=max(x,0) $$

![Activation Functions - HackMD](https://i.imgur.com/Rdsu9wG.png)

+ 변형으로도 쓰인다. 차이점은 양수만이 아닌 음수도 포함하는 것이다.
	$$PReLU(X) =max(0,X)+\alpha min(0,x)$$
	![Figure 7 | Cascading and Residual Connected Network for Single Image  Superresolution](https://static-01.hindawi.com/articles/wcmc/volume-2021/5579090/figures/5579090.fig.007.svgz)

## 2. Multiple Hidden Layer
앞서 본 single hidden layer와 같은 의미이나, hidden layer가 2개 이상일 경우 multiple hidden layer라고 한다.


$$input:x \\ 
 h_1= \sigma (w_{1}x +b_{1}) \\ 
   h_2= \sigma (w_{1}x +b_{1}) \\ 
   h_3= \sigma (w_{1}x +b_{1}) \\
   o=w_{4}^{T}h_3+b_{4}$$

hidden layer(은닉층)이 많아지면 복잡도가 증가하여 오버피팅(overfitting)이 될 가능성이 높다. 다음시간에 이러한 문제를 방지하기 위해 필요한 모델 선택과 관련하여 정리한다.

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTUwNTEwMDE0NV19
-->
