#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : CameraParameters.py                                      -->
#<!-- Description: Class used for managing the cameras parameters           -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 07/04/2015                                               -->
#<!-- Change     : 07/04/2015 - Creation of these classes                   -->
#<!--            : 07/12/2015 - Adapter for the new SIGB Framework          -->
#<!-- Review     : 26/04/2016 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2016042601 $"

########################################################################
import numpy as np
from scipy.linalg import qr

########################################################################
class CamerasParameters(object):
    """CamerasParameters Class is used for managing the cameras parameters."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Index(self):
        """Get the camera index."""
        return self.__Index

    @property
    def P(self):
        """Get the projection matrix."""
        return self.__P

    @P.setter
    def P(self, value):
        """Set the projection matrix."""
        self.__P = value

    @property
    def K(self):
        """Get the camera matrix."""
        return self.__K

    @K.setter
    def K(self, value):
        """Set the camera matrix."""
        self.__K = value

    @property
    def R(self):
        """Get the rotation matrix."""
        return self.__R

    @R.setter
    def R(self, value):
        """Set the rotation matrix."""
        self.__R = value

    @property
    def t(self):
        """Get the translation matrix."""
        return self.__t

    @t.setter
    def t(self, value):
        """Set the translation matrix."""
        self.__t = value

    @property
    def DistCoeffs(self):
        """Get the camera distortion coefficients."""
        return self.__DistCoeffs

    @DistCoeffs.setter
    def DistCoeffs(self, value):
        """Set the camera distortion coefficients."""
        self.__DistCoeffs = value

    @property
    def E(self):
        """Get the essential matrix."""
        return self.__E

    @E.setter
    def E(self, value):
        """Set the essential matrix."""
        self.__E = value

    @property
    def F(self):
        """Get the fundamental matrix."""
        return self.__F

    @F.setter
    def F(self, value):
        """Set the fundamental matrix."""
        self.__F = value

    @property
    def Q(self):
        """Get the disparity-to-depth mapping matrix."""
        return self.__Q

    @Q.setter
    def Q(self, value):
        """Set the disparity-to-depth mapping matrix."""
        self.__Q = value

    @property
    def Maps(self):
        """Get the undistortion and rectification transformation map."""
        return self.__Maps

    @Maps.setter
    def Maps(self, value):
        """Set the undistortion and rectification transformation map."""
        self.__Maps = value

    #----------------------------------------------------------------------#
    #                  CamerasParameters Class Constructor                 #
    #----------------------------------------------------------------------#
    def __init__(self, index):
        """CamerasParameters Class Constructor."""
        self.__Index = index
        self.Clear()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Center(self):
        """Compute and return the camera center."""
        _, R, t = self.__Factor()
        c = -np.dot(R.T, t)
        return c

    def Factor(self):
        """Factorize the camera matrix into K, R, t as P = K[R|t]."""
        P = np.matrix(self.P)

        if P.max() == 0:
            return np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 1))

        K, R = self.__RQ(P[:, :3])
        t = np.diag(np.sign(np.diag(K)))

        K = np.dot(K, t)
        R = np.dot(t, R)
        t = np.dot(np.linalg.inv(K), P[:, 3])

        return K, R, t

    def Project(self, point):
        """Project the input point based on a camera matrix."""
        if point.shape[0] != 4:
            point = np.vstack((point, np.ones((1, point.shape[1]))))

        point = np.dot(self.P, point)
        for i in range(3):
            point[i] /= point[2]

        return np.array(point.T)

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
    def __RQ(self, matrix):
        """Estimate the factor first 3*3 part."""
        Q, R = qr(np.flipud(matrix).T)
        R = np.flipud(R.T)
        Q = Q.T

        return R[:, ::-1], Q[::-1, :]

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def Clear(self):
        """Define the default values for all class attributes."""
        self.P = np.zeros((3, 4))
        self.K = np.zeros((3, 3))
        self.R = np.zeros((3, 3))
        self.t = np.zeros((3, 1))
        self.DistCoeffs = np.zeros((1, 5))
        self.E = np.zeros((3, 3))
        self.F = np.zeros((3, 3))
        self.Q = np.zeros((4, 4))
        self.Maps = [np.zeros((1, 1, 1)), np.zeros((1, 1, 1))]

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.VideoCaptureDevices.CameraParameters object."
