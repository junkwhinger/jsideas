---
layout:     post
title:      "The First and Second Derivatives and Local Minima"
date:       2017-12-31 00:00:00
author:     "Jun"
img: 20171231.png
tags: [python, math]
---

Whilst brushing up on linear algebra, I stumbled upon this <a href="https://math.dartmouth.edu/opencalc2/cole/lecture8.pdf">webpage</a> that provides a nice and short explanation on the first and second derivatives. To deepen my understanding I visualised the relationship between the first and second derivatives and local minimum / maximum.

### The Meaning of the First and Second Derivatives
- the function $$f(x)$$
- its first derivative $$f'(x)$$ is the slope of the tangent line at point x
  - it tells whether the function is increasing or decreasing and how much it is increasing or decreasing
  - if the first derivative is 0, x is called a critical point of $$f(x)$$
- its second derivative $$f''(x)$$
  - it tells if the first derivative is increasing or decreasing. when the second derivative is positive, the curve $$f(x)$$ is concave up, and vice versa.
  - when the second derivative is 0 then we do not know anything new about the behaviour of $$f(x)$$ at that point

### Critical Points and the Second Derivative Test
- we can use the second derivative to find out when x is a local maximum or minimum
- suppose that x is a critical point (first derivative = 0) and the second derivative is positive. 
  - the second derivative tells us that the first derivative is increasing at that point and the graph is concave up.
  - the only way to visualise this is local minimum where the slope of the function is zero but the graph is concave up. 
  - when the second derivative is negative with x being the critical point, it means that x is the local maximum.

### Visualisation with Python
Sympy is a great python library when playing with mathematical symbols and complex formulas. Taking a derivative is easy with Sympy.


```python
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

%matplotlib inline
```

Suppose $$f(x) = x^3 - 9x^2 + 15x - 7$$


```python
x = Symbol('x')
```


```python
y = x**3 - 9*x**2 + 15*x - 7
```


```python
## the first derivative
yfirst = y.diff(x)
print(yfirst)
```

>> 3*x**2 - 18*x + 15



```python
## the second derivative
ysecond = yfirst.diff(x)
print(ysecond)
```

>> 6*x - 18



```python
## input x range
test_x = np.linspace(-1, 7, 50)
```


```python
## corresponding y, first_derivative, second_derivative
test_y = [y.subs({x:v}) for v in test_x]
test_y_f = [yfirst.subs({x:v}) for v in test_x]
test_y_s = [ysecond.subs({x:v}) for v in test_x]
```


```python
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(test_x, test_y, color='black', label='function_f')
ax.plot(test_x, test_y_f, color='red', label='first_order')
ax.plot(test_x, test_y_s, color='blue', label='second_order')
ax.plot(test_x, np.zeros(len(test_x)), 'g--', label='y=0')

circle1 = plt.Circle((1, 0), 1, color='red', fill=False)
circle2 = plt.Circle((5, -32), 1, color='r', fill=False)

ax.add_artist(circle1)
ax.add_artist(circle2)

plt.annotate('local maximum', xy=(1, 0), xytext=(1, 2.5), fontsize=15)
plt.annotate('local minimum', xy=(5, -32), xytext=(5, -30), fontsize=15)

ax.legend(fontsize=15)
plt.show()
```


![png](/assets/materials/20171231/The%20First%20and%20Second%20Derivatives%20and%20Local%20Minima_10_0.png)


As stated above, the function is at local maximum when the first derivative is at 0 and the second derivative is negative. And it's at local minimum when the second derivative is positive. 
