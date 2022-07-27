from PIL import Image
import cv2
import imutils
from imutils import perspective
from imutils import contours
import numpy as np

class contoursGeometry:
    def __init__(self, img, img_final, length):
        self.img = img
        self.img_final = img_final
        self.length = length

    def _euclidean_distance(self, x1, x2):
        distances = ((np.asarray(x1)-np.asarray(x2))**2).sum(axis=-1)
        return np.sqrt(distances)
    
    def generate_image(self):
        cnts = cv2.findContours(np.uint8(self.img_final), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)

        # loop over the contours individually
        for c in cnts:
        # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 1:
                continue
        # compute the rotated bounding box of the contour
            orig = self.img.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 5)
            
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 8, (0, 0, 255), -1)


        (tl, tr, br, bl) = box
        (tltrX, tltrY) =  (tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5
        (blbrX, blbrY) =  (bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = (tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5
        (trbrX, trbrY) = (tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5

        # draw the midpoints on the image
        final = cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        final = cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        final = cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        final = cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        final = cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 5)
        final = cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 5)

        # compute the Euclidean distance between the midpoints
        dA = self._euclidean_distance((tltrX, tltrY), (blbrX, blbrY))
        dB = self._euclidean_distance((tlblX, tlblY), (trbrX, trbrY))

        pixelsPerMetric = dA / self.length

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # draw the object sizes on the image
        final = cv2.putText(orig, "{:.1f}mm".format(dimB), (int(tltrX), int(tltrY - 15)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 5)
        final = cv2.putText(orig, "{:.1f}mm".format(dimA), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 5)

        Image.fromarray(self.img).convert('RGB').save('img.png')
        Image.fromarray(final).convert('RGB').save('final.png')
