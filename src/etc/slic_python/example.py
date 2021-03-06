import numpy as np
import Image
import matplotlib.pyplot as plt
import slic

im = np.array(Image.open("tennis.png"))
region_labels = slic.slic_n(im, 100, 10)
contours = slic.contours(im, region_labels, 10)
plt.imshow(contours[:, :, :-1].copy())
plt.show()
