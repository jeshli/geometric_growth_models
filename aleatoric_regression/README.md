# Aleatoric Regression
A linear model which models both the mean and the variance. 
```math
Loss = \sum_{\forall i} \frac{(y_i - \mu(x_i))^2}{\sigma(x_i)^2} + \frac{1}{2}log(\sigma(x_i)^2)
```

## Demo 
The Starter_Code.py is prepared to input a DataFrame with a Date Column and a Cycles Burned column. It log transforms the Cycles Burned in order to produce an exponential function. It outputs graphs of the model and prints the model parameters.

## Dependencies
- Pytorch             https://pytorch.org/TensorRT/tutorials/installation.html
- Pandas              https://pandas.pydata.org/docs/getting_started/install.html
- Numpy               https://numpy.org/install/
- Matplotlib          https://matplotlib.org/stable/users/installing/index.html
