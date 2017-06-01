# Note

## AF and Linear function

- linear => y = Wx
- nonlinear => ?

- linear => nonlinear
=> y = Wx => y = AF(Wx)

- AF(): activation function
    - relu ![relu wiki](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Ramp_function.svg/488px-Ramp_function.svg.png)
    - sigmoid ![sigmoid wiki](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)
    - tanh ![tanh](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Sinh_cosh_tanh.svg/384px-Sinh_cosh_tanh.svg.png)
    - must differential

## Optimizer
- Stochastic Gradient Descent (SCG)
- Momentum
    - W += - Learning rate * dx
    - `m = b1 * m - Learning rate * dx`
      `W += m`
- AdaGrad
    - W += - Learning rate * dx
    - `v += dx^2`
      `W += - Learning rate * dx / v^(1/2)`
- RMSProp
    - W += - Learning rate * dx
    - Momentum + AdaGrad
    - `v = b1 * v + (1-b1) * dx^2`
      `W += -Learning rate * dx / v^(1/2)`
- Adam
    - W += - Learning rate * dx
    - `m = b1 * m + (1-b1) * dx`
      `v = b2 * v + (1-b2) * dx^2`
      `W += -Learning rate * m/ v^(1/2)`

