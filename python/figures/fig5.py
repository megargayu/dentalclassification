import matplotlib.pyplot as plt
import json
import os

# Path for data measured with Android Studio
# time = Time used to run on random image (seconds)
# ram = RAM used by model in memory (MB) 
# load = Maximum cpu usage when loading model (%)
# runImage = Maximum cpu usage when running random image through model (%)
data_path = os.path.join(os.path.dirname(__file__), "fig5.json")

# Load data
with open(data_path, "r") as fin:
  data = json.load(fin)

  timeData = data["time"]
  timeModels = list(timeData.keys())
  timeValues = list(timeData.values())

  ramsData = data["ram"]
  ramsModels = list(ramsData.keys())
  ramsValues = list(ramsData.values())

  loadModelData = data["load"]
  loadModelModels = list(loadModelData.keys())
  loadModelValues = list(loadModelData.values())

  runImageData = data["runImage"]
  runImageModels = list(runImageData.keys())
  runImageValues = list(runImageData.values())


# Plot the data!
fig, axs = plt.subplots(2, 2, figsize=(8, 6.5))

axs[0, 0].bar(timeModels, timeValues, color='lightblue', width = 0.4)
axs[0, 0].set_title("Seconds to Run On Image")
axs[0, 0].set(xlabel='Models', ylabel='Time (Seconds)')
for i in range(len(timeValues)):
  axs[0, 0].text(i, timeValues[i], timeValues[i], ha = 'center')

axs[0, 1].bar(ramsModels, ramsValues, color='lightblue', width = 0.4)
axs[0, 1].set_title('RAM Used')
axs[0, 1].set(xlabel='Models', ylabel='RAM (MB)')
for i in range(len(ramsValues)):
  axs[0, 1].text(i, ramsValues[i], ramsValues[i], ha = 'center')

axs[1, 0].bar(loadModelModels, loadModelValues, color='lightblue', width = 0.4)
axs[1, 0].set_title('CPU Usage During Model Loading')
axs[1, 0].set(xlabel='Models', ylabel='CPU Usage (Percentage)')
for i in range(len(loadModelValues)):
  axs[1, 0].text(i, loadModelValues[i], str(loadModelValues[i]) + "%", ha = 'center')

axs[1, 1].bar(runImageModels, runImageValues, color='lightblue', width = 0.4)
axs[1, 1].set_title('CPU Usage During Run on One Image')
axs[1, 1].set(xlabel='Models', ylabel='CPU Usage (Percentage)')
for i in range(len(runImageValues)):
  axs[1, 1].text(i, runImageValues[i], str(runImageValues[i]) + "%", ha = 'center')

fig.tight_layout()
plt.ioff()
plt.show()
