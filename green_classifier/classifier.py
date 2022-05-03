# %%
import numpy as np
import os
from PIL import Image

# %%
from sklearn.linear_model import LogisticRegression

# %%
rgb = []
label = []
for image_file in os.listdir('images/'):
    green = image_file[:3] != 'not'
    path = 'images/' + image_file
    im = Image.open(path)
    pic = im.load()
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            rgb.append(pic[x, y][0:3])
            label.append(green)

print(rgb[-2:])
    

# %%
clf = LogisticRegression(random_state=0).fit(rgb, label)

# %%
clf.score(rgb, label)

# %%
coeffs = clf.coef_

# %%
print(sig(np.dot(coeffs, rgb[-1])))

# %%
def sig(x):
    return 1/ (1 + np.exp(-x))

# %%



