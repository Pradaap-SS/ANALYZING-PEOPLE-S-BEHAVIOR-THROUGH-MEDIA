
import os
import numpy as np
import dlib
import cv2

from data import FaceData

#=============================================
class FaceDetector:
    

    _detector = None
    

    _predictor = None
    

    #---------------------------------------------
    def detect(self, image, downSampleRatio = None):
        

        #####################
        # Setup the detector
        #####################

        # Initialize the static detector and predictor if this is first use
        if FaceDetector._detector is None or FaceDetector._predictor is None:
            FaceDetector._detector = dlib.get_frontal_face_detector()

            faceModel = os.path.abspath('{}/models/face_model.dat' \
                            .format(os.path.dirname(__file__)))
            FaceDetector._predictor = dlib.shape_predictor(faceModel)

        #####################
        # Performance cues
        #####################

        # If requested, scale down the original image in order to improve
        # performance in the initial face detection
        if downSampleRatio is not None:
            detImage = cv2.resize(image, (0, 0), fx=1.0 / downSampleRatio,
                                                 fy=1.0 / downSampleRatio)
        else:
            detImage = image

        #####################
        # Face detection
        #####################

        # Detect faces in the image
        detectedFaces = FaceDetector._detector(detImage, 1)
        if len(detectedFaces) == 0:
            return False, None

        # No matter how many faces have been found, consider only the first one
        region = detectedFaces[0]

        # If downscaling was requested, scale back the detected region so the
        # landmarks can be proper located on the image in full resolution
        if downSampleRatio is not None:
            region = dlib.rectangle(region.left() * downSampleRatio,
                                    region.top() * downSampleRatio,
                                    region.right() * downSampleRatio,
                                    region.bottom() * downSampleRatio)

        # Fit the shape model over the face region to predict the positions of
        # its facial landmarks
        faceShape = FaceDetector._predictor(image, region)

        #####################
        # Return data
        #####################

        face = FaceData()

        # Update the object data with the predicted landmark positions and
        # their bounding box (with a small margin of 10 pixels)
        face.landmarks = np.array([[p.x, p.y] for p in faceShape.parts()])

        margin = 10
        x, y, w, h = cv2.boundingRect(face.landmarks)
        face.region = (
                       max(x - margin, 0),
                       max(y - margin, 0),
                       min(x + w + margin, image.shape[1] - 1),
                       min(y + h + margin, image.shape[0] - 1)
                      )

        return True, face