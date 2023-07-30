import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay

# Data
confusion_mobilenet = np.array([[4, 6], [0, 12]])
confusion_resnet = np.array([[9, 1], [3, 9]])

# Plot
fig, axs = plt.subplots(1, 2, figsize=(9, 3))

mobilenet_conf_matrix = ConfusionMatrixDisplay(confusion_mobilenet)
mobilenet_conf_matrix.plot(ax=axs[0])
axs[0].get_images()[0].colorbar.remove()
axs[0].set_title("MobileNet Confusion Matrix")

resnet_conf_matrix = ConfusionMatrixDisplay(confusion_resnet)
resnet_conf_matrix.plot(ax=axs[1])
axs[1].get_images()[0].set_clim(0, 12)
axs[1].set_title("ResNet Confusion Matrix")

fig.tight_layout()
plt.ioff()
plt.show()
