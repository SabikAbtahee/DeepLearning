# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\coins.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
#
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
#
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
#
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
#
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
#
# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]















#!/usr/bin/env python

'''
Watershed segmentation
=========
This program demonstrates the watershed segmentation algorithm
in OpenCV: watershed().
Usage
-----
watershed.py [image filename]
Keys
----
  1-7   - switch marker color
  SPACE - update segmentation
  r     - reset
  a     - toggle autoupdate
  ESC   - exit
'''


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from common import Sketcher

class App:
    def __init__(self, fn):
        self.img = cv.imread(fn)
        if self.img is None:
            raise Exception('Failed to load image file: %s' % fn)

        h, w = self.img.shape[:2]
        self.markers = np.zeros((h, w), np.int32)
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255

        self.auto_update = True
        self.sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors)

    def get_colors(self):
        return list(map(int, self.colors[self.cur_marker])), self.cur_marker

    def watershed(self):
        m = self.markers.copy()
        cv.watershed(self.img, m)
        overlay = self.colors[np.maximum(m, 0)]
        vis = cv.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv.CV_8UC3)
        cv.imshow('watershed', vis)

    def run(self):
        while cv.getWindowProperty('img', 0) != -1 or cv.getWindowProperty('watershed', 0) != -1:
            ch = cv.waitKey(50)
            if ch == 27:
                break
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print('marker: ', self.cur_marker)
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print('auto_update if', ['off', 'on'][self.auto_update])
            if ch in [ord('r'), ord('R')]:
                self.markers[:] = 0
                self.markers_vis[:] = self.img
                self.sketch.show()
        cv.destroyAllWindows()


if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\coins.jpg'
    print(__doc__)
    App(fn).run()
