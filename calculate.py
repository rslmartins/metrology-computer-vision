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
        cnts = cv2.findContours(np.uint8(self.img_final), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)

        centersE = []
        widthsE = []
        heightsE = []

        # loop over the contours individually
        for c in cnts:
            approx = cv2.approxPolyDP(c, .03 * cv2.arcLength(c, True), True)

            # if it is a triangle or pentagon
            if (len(approx) == 3) or (len(approx) == 5):
                continue

            # if it is either square or rectangle
            elif len(approx)==4:
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
                
            elif len(approx)==8:
                orig = self.img.copy()
                area = cv2.contourArea(c)
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                circleArea = radius * radius * np.pi
                if circleArea == area:
                    cv2.drawContours(orig, [c], 8, (0, 0, 255), 2)
                else:
                    ellipse = cv2.fitEllipse(c)
                    centerE = ellipse[0]
                    widthE = ellipse[1][0]
                    heightE = ellipse[1][1]
                    orig = cv2.ellipse(orig,ellipse,(0, 0, 255), 2)
                    if len(centersE)>0:
                        if (round(centerE[0],1) == round(centersE[-1][0], 1)) and (round(centerE[1],1) == round(centersE[-1][1], 1)):
                            continue
                        else:
                            centersE.append(centerE)
                            widthsE.append(widthE) 
                            heightsE.append(heightE)
                    else:
                        centersE.append(centerE)
                        widthsE.append(widthE) 
                        heightsE.append(heightE)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) =  (tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5
        (blbrX, blbrY) =  (bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = (tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5
        (trbrX, trbrY) = (tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5

        # compute the Euclidean distance between the midpoints
        dA = self._euclidean_distance((tltrX, tltrY), (blbrX, blbrY))
        dB = self._euclidean_distance((tlblX, tlblY), (trbrX, trbrY))

        pixelsPerMetric = dA / self.length

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # draw the object sizes on the image
        final = cv2.putText(orig, "{:.1f}mm".format(dimB), (int(tltrX), int(tltrY - 15)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 5)
        final = cv2.putText(orig, "{:.1f}mm".format(dimA), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 5)
        for i in range(len(widthsE)):
            final = cv2.putText(orig, "{}, {} mm".format(round(widthsE[i],2), round(heightsE[i],2)), (int(centersE[i][0] - 45), int(centersE[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        Image.fromarray(self.img).convert('RGB').save('img.png')
        Image.fromarray(final).convert('RGB').save('final.png')
