from copy import deepcopy
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils

imp_url = 'http://snappygoat.com/b/5e3c07f2560b6420543d8e2367a70d3cdb08c39e'
img = imutils.url_to_image(img_url)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgshow(img)