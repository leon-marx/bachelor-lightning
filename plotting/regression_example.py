import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(17)

x = sorted(np.random.normal(0, 50, 100))
y = sorted(np.random.normal(0, 50, 100) * 3)

x_eps = np.random.randn(100) * 20
y_eps = np.random.randn(100) * 20

x += x_eps + 300
y += y_eps + 300

xerr = np.ones(100) * 2
yerr = np.ones(100) + 10

def linear(x, a, b):
    return a * x + b
xx = np.linspace(x.min(), x.max(), 200)

popt, pcov = curve_fit(linear, x, y, sigma=yerr)

plt.figure(figsize=(12, 8))
plt.errorbar(x, y, xerr=xerr, yerr=yerr, color="k", fmt="none", label="Noisy Data")
plt.plot(xx, linear(xx, *popt), label=f"Linear Fit", color="C1")
plt.plot()
plt.legend(fontsize=16)
plt.xticks([])
plt.yticks([])
plt.show()
