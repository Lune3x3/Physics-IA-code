import colour
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

vidPath = 'assets/diskCrpStr.mov'
frameOffset = 0
deg2frame = 0.015

def get_frame(fileNm, index):
    cap = cv.VideoCapture(fileNm)
    cap.set(1, index)
    ret, frame = cap.read()
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return img

def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()
    inp = input('list value: ')
    return int(inp)

def palette(clusters):
    width=300
    palette = np.zeros((50, width, 3), np.uint8)
    steps = width/clusters.cluster_centers_.shape[0]
    for idx, centers in enumerate(clusters.cluster_centers_):
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette

def rgb2wav(clr):
    r = clr[0]
    g = clr[1]
    b = clr[2]

    RGB_f = np.array([r, g, b]) / 255

    # Using the individual definitions:
    RGB_l = colour.models.eotf_sRGB(RGB_f)
    XYZ = colour.RGB_to_XYZ(
        RGB_l,
        colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
        colour.models.RGB_COLOURSPACE_sRGB.whitepoint,
        colour.models.RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ,
    )
    xy = colour.XYZ_to_xy(XYZ)
    wl, xy_1, xy_2 = colour.dominant_wavelength(
        xy, colour.models.RGB_COLOURSPACE_sRGB.whitepoint
    )

    # Using the automatic colour conversion graph:
    wl, xy_1, xy_2 = colour.convert(RGB_f, "Output-Referred RGB", "Dominant Wavelength")

    #colour.plotting.colour_style()
    #figure, axes = colour.plotting.plot_chromaticity_diagram_CIE1931(
    #    diagram_opacity=0.15, standalone=False
    #)
    #
    #xy_i = np.vstack([xy_1, xy, colour.models.RGB_COLOURSPACE_sRGB.whitepoint, xy_2])
    #axes.plot(xy_i[..., 0], xy_i[..., 1], "-o")
    #colour.plotting.render()
    return wl

dim = (500, 300)

with open('dataPoints.txt', 'a') as f:
    for i in range(100):
        img = get_frame(vidPath, i*30)
        img = cv.resize(img, dim, interpolation = cv.INTER_NEAREST)

        clt = KMeans(n_clusters=20)
        clt.fit(img.reshape(-1, 3))

        val = show_img_compar(img, palette(clt))

        wav = rgb2wav(clt.cluster_centers_[val])

        incline = (i*30) * deg2frame

        f.write(str(wav) + ' : ' + str(incline) + '\n')

