
import cv2
from collections import OrderedDict
import numpy as np
count=0
#=============================================
class FaceData:
    """
    Represents the data of a face detected on an image.
    """

    _jawLine = [i for i in range(17)]
    """
    Indexes of the landmarks at the jaw line.
    """

    _rightEyebrow = [i for i in range(17,22)]
    """
    Indexes of the landmarks at the right eyebrow.
    """

    _leftEyebrow = [i for i in range(22,27)]
    """
    Indexes of the landmarks at the left eyebrow.
    """

    _noseBridge = [i for i in range(27,31)]
    """
    Indexes of the landmarks at the nose bridge.
    """

    _lowerNose = [i for i in range(30,36)]
    """
    Indexes of the landmarks at the lower nose.
    """

    _rightEye = [i for i in range(36,42)]
    """
    Indexes of the landmarks at the right eye.
    """

    _leftEye = [i for i in range(42,48)]
    """
    Indexes of the landmarks at the left eye.
    """

    _outerLip = [i for i in range(48,60)]
    """
    Indexes of the landmarks at the outer lip.
    """

    _innerLip = [i for i in range(60,68)]
    """
    Indexes of the landmarks at the inner lip.
    """

    #---------------------------------------------
    def __init__(self, region = (0, 0, 0, 0),
                 landmarks = [0 for i in range(136)]):
        

        self.region = region
        

        self.landmarks = landmarks
        

    #---------------------------------------------
    def copy(self):
        
        return FaceData(self.region, self.landmarks.copy())

    #---------------------------------------------
    def isEmpty(self):
        
        return all(v == 0 for v in self.region) or \
               all(vx == 0 and vy == 0 for vx, vy in self.landmarks)

    #---------------------------------------------
    def crop(self, image):
        
        left = self.region[0]
        top = self.region[1]
        right = self.region[2]
        bottom = self.region[3]

        croppedImage = image[top:bottom+1, left:right+1]

        croppedFace = self.copy()
        croppedFace.region = (0, 0, right - left, bottom - top)
        croppedFace.landmarks = [[p[0]-left, p[1]-top] for p in self.landmarks]

        return croppedImage, croppedFace

    #---------------------------------------------
    def draw(self, image, drawRegion = None, drawFaceModel = None):
        
        if self.isEmpty():
            raise RuntimeError('Can not draw the contents of an empty '
                               'FaceData object')

        # Check default arguments
        if drawRegion is None:
            drawRegion = True
        if drawFaceModel is None:
            drawFaceModel = True

        # Draw the region if requested
        if drawRegion:
            cv2.rectangle(image, (self.region[0], self.region[1]),
                                 (self.region[2], self.region[3]),
                                 (0, 0, 255), 2)

        # Draw the positions of landmarks
        color = (0, 255, 255)
        for i in range(68):
            cv2.circle(image, tuple(self.landmarks[i]), 1, color, 2)

        # Draw the face model if requested
        if drawFaceModel:
            c = (0, 255, 255)
            p = np.array(self.landmarks)

            cv2.polylines(image, [p[FaceData._jawLine]], False, c, 2)
            cv2.polylines(image, [p[FaceData._leftEyebrow]], False, c, 2)
            cv2.polylines(image, [p[FaceData._rightEyebrow]], False, c, 2)
            cv2.polylines(image, [p[FaceData._noseBridge]], False, c, 2)
            cv2.polylines(image, [p[FaceData._lowerNose]], True, c, 2)
            cv2.polylines(image, [p[FaceData._leftEye]], True, c, 2)
            cv2.polylines(image, [p[FaceData._rightEye]], True, c, 2)
            cv2.polylines(image, [p[FaceData._outerLip]], True, c, 2)
            cv2.polylines(image, [p[FaceData._innerLip]], True, c, 2)

        return image

#=============================================
class GaborData:
    """
    Represents the responses of the Gabor bank to the facial landmarks.
    """

    #---------------------------------------------
    def __init__(self, features = [0.0 for i in range(2176)]):
        
        self.features = features
        

    #---------------------------------------------
    def copy(self):
        
        return GaborData(self.features.copy())

    #---------------------------------------------
    def isEmpty(self):
        
        return all(v == 0 for v in self.features)