
import os
from collections import OrderedDict
import numpy as np

from gabor import GaborBank
from data import FaceData
from faces import FaceDetector

from sklearn import svm
from sklearn.externals import joblib

#=============================================
class InvalidModelException(Exception):
    '''
    Exception indicating that the detection model could not be loaded (or didn't
    exist).
    '''
    pass

#=============================================
class EmotionsDetector:
    """
    Implements the detector of prototypic emotions on face images.
    """

    #---------------------------------------------
    def __init__(self):
        """
        Class constructor.
        """

        self._clf = svm.SVC(kernel='rbf', gamma=0.001, C=10,
                            decision_function_shape='ovr',
                            probability=True, class_weight='balanced')
        

        self._emotions = OrderedDict([
                             (0, 'neutral'), (1, 'happiness'), (2, 'sadness'),
                             (3, 'anger'), (4, 'fear'),  (5, 'surprise'),
                             (6, 'disgust')
                         ])
#        self._count=0
    
        """
        Class and labels of the prototypic emotions detected by this model.
        """

        modulePath = os.path.dirname(__file__)
        self._modelFile = os.path.abspath('{}/models/emotions_model.dat' \
                            .format(modulePath))
        """
        Name of the file used to persist the model in the disk.
        """

        # Load the model from the disk, if its file exists
        if not os.path.isfile(self._modelFile):
            raise InvalidModelException('Could not find model file: {}' \
                    .format(self._modelFile))

        if not self.load():
            raise InvalidModelException('Could not load model from file: {}' \
                    .format(self._modelFile))

    #---------------------------------------------
    def load(self):
        

        try:
            clf = joblib.load(self._modelFile)
        except:
            return False

        self._clf = clf
        return True

    #---------------------------------------------
    def _relevantFeatures(self, gaborResponses, facialLandmarks):
        

        # Get the 32 responses at the positions of all the face landmarks
        points = np.array(facialLandmarks)

        # Try to get the responses for all points. If an exception is caught,
        # it is because some landmarks are out of the image area (i.e. the face
        # is partially occluded, but it was still possible to detect). In this
        # case, assume 0.0 for the responses of the landmarks outside the image
        # area.
        try:
            responses = gaborResponses[:, points[:, 1], points[:, 0]]
        except:
            w = gaborResponses.shape[2]
            h = gaborResponses.shape[1]

            responses = np.zeros((32, 68), dtype=float)
            for i in range(len(points)):
                x = points[i][0]
                y = points[i][1]
                if x < w and y < h:
                    responses[:, i] = gaborResponses[:, y, x]
                else:
                    responses[:, i] = 0.0

        # Reshape the bi-dimensional matrix to a single dimension
        featureVector = responses.reshape(-1).tolist()

        return featureVector

    #---------------------------------------------
    def detect(self, face, gaborResponses):
        

        # Filter only the responses at the facial landmarks
        features = self._relevantFeatures(gaborResponses, face.landmarks)

        # Return the prediction based on the given features
        return self.predict(features)

    #---------------------------------------------
    def predict(self, features):
        

        # Predict the emotion probabilities on the given features
        probas = self._clf.predict_proba([features])[0]

        # Build a dictionary with the probabilities and emotion labels
        ret = OrderedDict()
        for i in range(len(self._emotions)):
            label = self._emotions[i]
            ret[label] = probas[i]

        return ret