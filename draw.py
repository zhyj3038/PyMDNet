import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def draw(path, bb):
  im = np.array(Image.open(path), dtype=np.uint8)
  fig,ax = plt.subplots(1)
  ax.imshow(im)
  rect = patches.Rectangle(bb, im.size[0], im.size[1],linewidth=1,edgecolor='r',facecolor='none'))
  ax.add_patch(rect)
  plt.show()
  

