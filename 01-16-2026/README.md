# Mạng `neural` 1 lớp ẩn 2 đầu vào, hai node ẩn và 2 đầu ra

Hàm kích hoạt: `sigmoid(x)` = $\frac{1}{1+e^{-x}}$

## Tổng trọng số cho nút ẩn H1 và H2

$$
H_1 = (w_{1}*x_{1}+w_{2}*x_{2}+b_{1})
$$

$$
H_2 = (w_{3}*x_{1}+w_{4}*x_{2}+b_{1})
$$

## Đầu ra cho nút ẩn H1 và H2

$$
h1f = sigmoid(H_1) = \frac{1}{1+e^{-H_{1}}}
$$

$$
h2f = sigmoid(H_2) = \frac{1}{1+e^{-H_{2}}}
$$

## Tổng trọng số cho đầu ra y1 và y2

$$
y_{1} = h1f * w_{5}+h2f*w_{6}+b_{2}
$$

$$
y_2 = h1f * w_{7}+h2f*w_{8}+b_{2}
$$

## Đầu ra cho tại y1 và y2

$$
y_{1final} = sigmoid(y_{1}) = \frac{1}{1+e^{-y_{1}}}
$$

$$
y_{2final} = sigmoid(y_{2}) = \frac{1}{1+e^{-y_{2}}}
$$

## Hàm mất mát

$$
Loss = \frac{1}{2}*[(y_{1final}-x_{1})^2+(y_{2final}-x_{2})^2]
$$

## Đạo hàm tại đầu ra

```cpp
double dy1 = (y1 - x1_d) * y1 * (1 - y1);
double dy2 = (y2 - x2_d) * y2 * (1 - y2);
```

Công thức là:

$$
\delta_{y1} = \frac{\delta{L}}{\delta_{y1_{final}}} * \frac{\delta{y1_{final}}}{\delta{y_{1}}} = (y1_{final}-x_{1}) * y1_{final} * (1-y1_{final})
$$
$$
\delta_{y_{2}} = \frac{\delta{L}}{\delta_{y_2final}} * \frac{\delta{y_{2final}}}{\delta{y_{2}}} = (y_{2final}-x_{2}) * y_{2final}*(1-y_{2final})
$$

$$
\frac{\delta{L}}{\delta{w_{y_{i}}}}=\delta_{y_{i}}*x_{i}
$$



## Đạo hàm tại lớp ẩn

```cpp
double dh1 = (dy1*w[5] + dy2*w[7]) * h1f * (1 - h1f);
double dh2 = (dy1*w[6] + dy2*w[8]) * h2f * (1 - h2f);
```

Công thức đạo hàm các trọng số cho nút xuất H1:

$$
\frac{\delta{L}}{\delta{wi}} = (y1_{final} - x_{1}) * (1-y1_{final}) * y1_{final} * w_{5} * (1-H1_{final}) * H1_{final} * x_{1} + \\
(y2_{final} - x_{2}) * (1-y2_{final}) * y2_{final} * w_{7} * (1-H1_{final}) * H1_{final} * x_{1}
$$

$$
 => (dy_{1} * w_{5} + dy_{2} * w_{7}) * h1f * (1-h1f) * x_{i}
$$



Công thức đạo hàm cho các trọng số cho nút xuất H2:

$$
\frac{\delta{L}}{\delta{w_{i}}}=(dy_{1}*w_{6}+dy_{2}*w_{8})*h2f*(1-h2f)*x_{i}
$$



## Cập nhật trọng số cho đầu ra

Công thức:

$$
w_{i-new} = w_{i} - \eta* \frac{\delta{L}}{\delta{w_{i}}}
$$

với $\eta$ là tốc độ học (learning rate)

$$
w_{i-new} = w_{i} - \eta* \frac{\delta{L}}{\delta{w_{i}}}
$$

$ với  \eta$ là tốc độ học (learning rate)

## Cập nhật số bias

$$
b_1 = \eta(dh1+dh2) \\
b_2 = \eta(dy1+dy2) 
$$