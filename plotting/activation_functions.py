import torch
import matplotlib.pyplot as plt


relu = torch.nn.ReLU()
lrelu = torch.nn.LeakyReLU()
elu = torch.nn.ELU()
selu = torch.nn.SELU()

activations = [relu, lrelu, elu, selu]
activation_names = ["ReLU", "LReLU", "ELU", "SELU"]
long_activation_names = ["Rectified Linear Unit", "Leaky Rectified Linear Unit", "Exponential Linear Unit", "Scaled Exponential Linear Unit"]

xx = torch.linspace(-6, 6, 1000)

plt.figure(figsize=(16, 16))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.plot(xx, activations[i](xx), color=f"C{i}", linewidth=2)
    plt.xlim(-6.5, 6.5)
    plt.ylim(-2.5, 6.5)
    plt.title(f"{long_activation_names[i]}", size=18)
plt.show()
