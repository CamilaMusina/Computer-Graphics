import numpy as np
#import cv2
import argparse
from PIL import Image

def sobel(img, threshold):
    G_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    G_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    temp1 = np.array(img)
    mag = np.zeros(temp1.shape)

    temp = temp1
    temp = np.append([temp1[0]], temp, axis=0)
    temp = np.append(temp, [temp[-1]], axis=0)
    l = []
    for i in range(temp.shape[0]):
        l.append([temp[i][0]])
    npl = np.array(l)
    r = []
    for i in range(temp.shape[0]):
        r.append([temp[i][-1]])
    npr = np.array(r)
    temp = np.append(npl, temp, axis=1)
    temp = np.append(temp, npr, axis=1)

    for y in range(0, rows - 2):
        for x in range(0, columns - 2):
            z1 = temp[y+1][x + 1]
            z2 = temp[y+1][x + 2]
            z3 = temp[y+1][x + 3]

            z4 = temp[y + 2][x + 1]
            z5 = temp[y+2][x+2]
            z6 = temp[y+2][x + 3]

            z7 = temp[y + 3][x + 1]
            z8 = temp[y + 3][x+2]
            z9 = temp[y + 3][x + 3]

            v = z7 + 2*z8 + z9 - (z1 + 2*z2 + z3)
            h = z3 + 2*z6 + z9 - (z1 + 2*z4 + z7)
            mag[y+1, x+1] = np.sqrt((v ** 2) + (h ** 2))
            
    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0
    new_image = Image.fromarray(mag)
    new_p = new_image.convert("L")
    new_p.save("sobel.jpg")
   # cv2.imshow("Image", new_image)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()


img = Image.open(r"path to file")
img1 = img.convert("L")
image = sobel(img1, 70)

#img = cv2.imread('image.png', 0)  # read an image
#mag = sobel(img, 70)
#a = sub_plot(img, mag)
#img_show(a)
