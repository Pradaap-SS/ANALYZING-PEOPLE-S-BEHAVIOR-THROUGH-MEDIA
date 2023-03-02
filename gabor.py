
import numpy as np
from skimage.filters import gabor_kernel
import cv2

#=============================================
class KernelParams:
    """
    A simple class to represent the parameters of a given Gabor kernel.
    """

    #---------------------------------------------
    def __init__(self, wavelength, orientation):
        

        self.wavelength = wavelength
        """Wavelength (in pixels) of a Gabor kernel."""

        self.orientation = orientation
        """Orientation (in radians) of a Gabor kernel."""

    #---------------------------------------------
    def __hash__(self):
        
        return hash((self.wavelength, self.orientation))

    #---------------------------------------------
    def __eq__(self, other):
        
        
        return (self.wavelength, self.orientation) == \
               (other.wavelength, other.orientation)

    #---------------------------------------------
    def __ne__(self, other):
        
        return not(self == other)

#=============================================
class GaborBank:
    """
    Represents a bank of gabor kernels.
    """

    #---------------------------------------------
    def __init__(self, w = [4, 7, 10, 13],
                       o = [i for i in np.arange(0, np.pi, np.pi / 8)]):
        

        self._wavelengths = w
        

        self._orientations = o
        

        self._kernels = {}
        

        # Create one kernel for each combination of wavelength x orientation
        for wavelength in self._wavelengths:
            for orientation in self._orientations:
                
                frequency = 1 / wavelength

                # Create and save the kernel
                kernel = gabor_kernel(frequency, orientation)
                par = KernelParams(wavelength, orientation)
                self._kernels[par] = kernel

    #---------------------------------------------
    def filter(self, image):
        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        responses = []
        for wavelength in self._wavelengths:
            for orientation in self._orientations:

                # Get the kernel
                frequency = 1 / wavelength
                par = KernelParams(wavelength, orientation)
                kernel = self._kernels[par]

                # Filter with both real and imaginary parts
                real = cv2.filter2D(image, cv2.CV_32F, kernel.real)
                imag = cv2.filter2D(image, cv2.CV_32F, kernel.imag)

                # The response is the magnitude of the real and imaginary
                # responses to the filters, normalized to [-1, 1]
                mag = cv2.magnitude(real, imag)
                cv2.normalize(mag, mag, -1, 1, cv2.NORM_MINMAX)

                responses.append(mag)

        return np.array(responses)