# Mạng neural 3 tầng ẩn

## Thông số ban đầu và kiến trúc mạng

- Đầu vào: $x_1$ = 0.1, $x_2$ = 0.2, $x_3$ = 0.3
- Đầu ra: $y_1$ =1, $y_2$ = 2, $y_3$ = 5

`Kiến trúc`:

```bash
x (3 input)
 → Hidden 1 (tanh, 3 node)
 → Hidden 2 (ReLU, 3 node)
 → Hidden 3 (SELU, 3 node)
 → Output (linear, MSE loss)
```

> Các trọng số ban đầu là tự do chọn

## Các hàm kích hoạt

### 1. tanh

Định nghĩa

$$
\frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

Đạo hàm

$$
\frac{d}{dx}tanh(x)=1-tanh^{2}(x)
$$


Khi có output: y = tanh(x) thì ta sẽ có được:

$$
\frac{dy}{dx} =1 - y^2
$$

### 2.ReLu

Định nghĩa

$$
ReLU(x) = max(0,x);
$$


Đạo hàm
$$
ReLU^{'}(x) = 
\begin{cases}
0, & x < 0 \\
1, & x > 0
\end{cases}
$$

### 3.SELU

Định nghĩa
$$
SELU(x)= \begin{cases}
\lambda x, & x>0 \\
\lambda \alpha (e^x - 1), & x \le 0
\end{cases}
$$
Trong đó ta có hai hằng số cố định là:

- $\alpha \approx 1.67326$
- $\lambda \approx 1.0507$

Đạo hàm
$$
SELU^{'}(x) = \begin{cases}
\lambda & x > 0 \\
\lambda \alpha 3^x & x \le 0
\end{cases}
$$

## Tổng trọng số tại một node

$$ H = \sum_{i=1}^{n} w_i x_i  + bias$$

## Đầu ra sau tại node bất kì

$$h_{final} = af(H) $$
Tuy nhiên vì đầu ra là kết nối linear tức là

```bash
H3_final[1] -> out[1]
H3_final[2] -> out[2]
H3_final[3] -> out[3]
```

Nên tính đầu ra sẽ đơn giản là:
$$ out_i = w_i x_i  + bias_i$$

## Hàm mất mát (MSE)

$$L = \frac{1}{2}[(out_1 - y_1)^2 + (out_2 - y_2)^2 + (out_3 - y_3)^2] $$

## Lan Truyền ngược

Công thức cốt lõi như sau:
$$
\delta = \frac{\partial L}{\partial z} = \frac{\partial L}{\partial a}*\frac{\partial a}{\partial z}
$$
Trong đó:

- z : Tổng trọng số (sum)
- a: output sau activation 
  
  ### 1. Tại output layer
  
  $$
  \delta^{out}_i = (y_{pred}-y_i)
  $$
  
  ### 2. Tại tầng ẩn số 3 (SELU)
  
  $$
  \delta^{3}_i = (\sum \delta^{out}_i * w^{out}_{ij}). SELU^{'}(z^{(3)}_j)
  $$
  
  ### 3. Tại tầng ẩn số 2 (ReLU)
  
  $$
  \delta^{2}_i = (\sum_k \delta^{3}_i * w^{3}_{ij}). ReLU^{'}(z^{(2)}_j)
  $$
  
  ### 4. Tại tầng ẩn số 1 (tanh)
  
  $$
  \delta^{1}_i = (\sum_k \delta^{2}_i * w^{2}_{ij}).(1- h_j^2)
  $$
  
  ## Cập nhật trọng số:
