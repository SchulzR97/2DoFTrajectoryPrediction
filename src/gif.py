import imageio
import os

images = []
for dir, _, files in os.walk('plots/gif'):
    files.sort()
    for file in files:
        if not file.endswith('.png'):
            continue
        images.append(imageio.imread(dir + '/' + file))
imageio.mimsave('plots/gif/readout.gif', images, duration=100)