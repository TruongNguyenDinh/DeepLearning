# Mạng `neural` 1 lớp ẩn 2 đầu vào, hai node ẩn và 2 đầu ra

Hàm kích hoạt: `sigmoid(x)` = $\frac{1}{1+e^{-x}}$

## Tổng trọng số cho nút ẩn H1 và H2

$$
H_1 = (w_1*x_1+w_2*x_2+b_1)
$$

$$
H_2 = (w_4*x2+w_4*x_2+b1)
$$

## Đầu ra cho nút ẩn H1 và H2

$$
h1f = sigmoid(H_1) = \frac{1}{1+e^{-H_1}}
$$

$$
h2f = sigmoid(H_2) = \frac{1}{1+e^{-H_2}}
$$

## Tổng trọng số cho đầu ra y1 và y2

$$
y_1 = h1f*w_5+h2f*w_6+b_2
$$

$$
y_2 = h1f*w_7+h2f*w_8+b_2
$$

## Đầu ra cho tại y1 và y2

$$
y_{1final} = sigmoid(y_1) = \frac{1}{1+e^{-y_1}}
$$

$$
y_{2final} = sigmoid(y_2) = \frac{1}{1+e^{-y_2}}
$$

## Hàm mất mát

$$
Loss = \frac{1}{2}*[(y_{1final}-x_1)^2+(y_{2final}-x_2)^2]
$$

## Đạo hàm tại đầu ra

```cpp
double dy1 = (y1 - x1_d) * y1 * (1 - y1);
double dy2 = (y2 - x2_d) * y2 * (1 - y2);
```

Công thức là:

$$
\delta_{y_1} = \frac{\delta{L}}{\delta_{y_1final}} * \frac{\delta{y_{1final}}}{\delta{y_1}} = (y_{1final}-x_{1})*y_{1final}*(1-y_{1final}) \\
\delta_{y_2} = \frac{\delta{L}}{\delta_{y_2final}} * \frac{\delta{y_{2final}}}{\delta{y_2}} = (y_{2final}-x_{2})*y_{2final}*(1-y_{2final})
$$

$$
\frac{\delta{L}}{\delta{w_{y_i}}}=\delta_{y_i}*x_i
$$



## Đạo hàm tại lớp ẩn

```cpp
double dh1 = (dy1*w[5] + dy2*w[7]) * h1f * (1 - h1f);
double dh2 = (dy1*w[6] + dy2*w[8]) * h2f * (1 - h2f);
```

Công thức đạo hàm các trọng số cho nút xuất H1:

$$
\frac{\delta{L}}{\delta{w_i}}=(y_{1final} - x_1)*(1-y_{1final})*y_{1final}*w_5*(1-H_{1final})*H_{1final}*x_1 \\
 + (y_{2final} - x_2)*(1-y_{2final})*y_{2final}*w_7*(1-H_{1final})*H_{1final}*x_1 \\
 => (dy_1*w_5 + dy_2*w_7)* h1f *(1-h1f)*x_i
$$



Công thức đạo hàm cho các trọng số cho nút xuất H2:

$$
\frac{\delta{L}}{\delta{w_i}}=(dy_1*w_6+dy_2*w_8)*h2f*(1-h2f)*x_i
$$



## Cập nhật trọng số cho đầu ra

Công thức:
$$
w_{i-new} = w_i - \eta* \frac{\delta{L}}{\delta{w_i}}
$$
với $\eta$ là tốc độ học (learning rate)

$$
w_{i-new} = w_i - \eta* \frac{\delta{L}}{\delta{w_i}}
$$

$ với  \eta$ là tốc độ học (learning rate)

## Cập nhật số bias

$$
b_1 = \eta(dh1+dh2) \\
b_2 = \eta(dy1+dy2) 
$$