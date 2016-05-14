#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : CalibrationManager.py                                    -->
#<!-- Description: Class used for managing the calibration process          -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 09/04/2015                                               -->
#<!-- Change     : 09/04/2015 - Creation of these classes                   -->
#<!--            : 07/12/2015 - Change the class for the new SIGB Framework -->
#<!-- Review     : 26/04/2016 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2016042601 $"

########################################################################
import cv2
import numpy as np

from ClassProperty import ClassProperty
from Pattern       import Pattern

from Framework.VideoCaptureDevices.CaptureManager import CaptureManager

########################################################################
class CalibrationManager(object):
    """CalibrationManager Class is used for managing the calibration process."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __Instance = None

    #----------------------------------------------------------------------#
    #                         Static Class Methods                         #
    #----------------------------------------------------------------------#
    @ClassProperty
    def Instance(self):
        """Create an instance for the calibration manager."""
        if self.__Instance is None:
            self.__Instance = Calibration()
        return self.__Instance

    #----------------------------------------------------------------------#
    #                    ImageManager Class Constructor                    #
    #----------------------------------------------------------------------#
    def __init__(self):
        """This constructor is never used by the system."""
        pass

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.Calibration.CalibrationManager object."


########################################################################
class Calibration(object):
    """Calibration Class is used for calibrating the connected cameras."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __path = "./Framework/VideoCaptureDevices/CalibrationData/"

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def IsCalibrated(self):
        """Check if the cameras are calibrated."""
        return self.__isCalibrated

    #----------------------------------------------------------------------#
    #                     Calibration Class Constructor                    #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Calibration Class Constructor."""
        self.__isCalibrated = False
        self.__Pattern = Pattern()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Calibrate(self):
        """Calibrate all connected cameras."""
        if CaptureManager.Instance.IsCalibrated():
            return

        # Vectors used by the calibration process.
        objectPoints = {}
        imagePoints  = {}
        cameraMatrix = {}
        distCoeffs   = {}
        for parameter in CaptureManager.Instance.Parameters:
            index = parameter.Index
            objectPoints[index] = []
            imagePoints[index]  = []
            cameraMatrix[index] = []
            distCoeffs[index]   = []

        # Get points from the pattern.
        patternPoints = self.__Pattern.CalculatePattern()

        # Define the number of chessboard that you want to use during the calibration process.
        N = 9

        # Number of detected chessboard.
        j = 0

        # While condition used for calibrating the camera.
        corners = {}
        while j < N:
            # Read the current image from a camera.
            images = CaptureManager.Instance.Read()
            for image, parameter in zip(images, CaptureManager.Instance.Parameters):
                # Get the camera parameter.
                index = parameter.Index

                # Get the image size.
                h, w  = image.shape[:2]
                chessboard = image.copy()

                # Finds the positions of internal corners of the chessboard.
                corner = self.__Pattern.FindCorners(chessboard)
                if corner is not None:
                    corners[index] = corner

                # Show the final processed image.
                cv2.imshow("Camera" + str(index) + "_Uncalibrated", chessboard)

                # Checks the keyboard button pressed by the user.
                ch = cv2.waitKey(1)
                if ch == ord("q"):
                    break
                elif ch == 32: # Press space key for taking the sample images.
                    if len(corners) == len(CaptureManager.Instance.Parameters):
                        j += 1
                        for key in corners:
                            # Saves the detected chessboard.
                            cv2.imwrite(self.__path + "Camera_" + str(key) + "_chessboard" + str(j) + ".png", image)

                            # Add the detected points in the vectors.
                            imagePoints[key].append(corners[key].reshape(-1, 2))
                            objectPoints[key].append(patternPoints)

                        corners = {}
                        # Long wait for showing to the user the selected chessboard.
                        cv2.waitKey(1000)

        # Close all allocated resources.
        cv2.destroyAllWindows()

        # Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
        if j > 0:
            # Calibrate each connected camera.
            for parameter in CaptureManager.Instance.Parameters:
                index = parameter.Index
                cameraMatrix[index], distCoeffs[index] = self.CalibrateCamera(index, imagePoints[index], objectPoints[index], (w, h))

            # Checks if it is necessary to undistort the image.
            isUndistorting = True

            # While condition used for testing the calibration.
            while True:
                # Read the current image from a camera.
                images = CaptureManager.Instance.Read()

                for image, parameter in zip(images, CaptureManager.Instance.Parameters):
                    # Get the camera index.
                    index = parameter.Index

                    # Transforms an image to compensate for lens distortion.
                    if isUndistorting:
                        image = cv2.undistort(image, cameraMatrix[index], distCoeffs[index])

                        # Show the final processed image.
                        cv2.imshow("Camera" + str(index) + "_Calibrated", image)

                # Checks the keyboard button pressed by the user.
                ch = cv2.waitKey(1)
                if ch == ord("q"):
                    break
                elif ch == ord("u"):
                    isUndistorting = not isUndistorting

            # Define the new calibration status.
            self.__isCalibrated = True

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()

    def CalibrateCamera(self, index, imagePoints, objectPoints, size):
        """Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern."""
        # Output 3x3 floating-point camera matrix and output vector of distortion coefficients.
        cameraMatrix = np.zeros((3, 3))
        distCoeffs   = np.zeros((5, 1))
    
        # Calibrates a single camera.
        _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, size, cameraMatrix, distCoeffs)

        # Save calibration process data.
        np.save(self.__path + "Camera_" + str(index) + "_cameraMatrix", cameraMatrix)
        np.save(self.__path + "Camera_" + str(index) + "_distCoeffs", distCoeffs)
        np.save(self.__path + "Camera_" + str(index) + "_rvecs", rvecs)
        np.save(self.__path + "Camera_" + str(index) + "_tvecs", tvecs)
        np.save(self.__path + "Camera_" + str(index) + "_img_points", imagePoints)
        np.save(self.__path + "Camera_" + str(index) + "_obj_points", objectPoints)

        # Return the final result
        return cameraMatrix, distCoeffs

    def CalibrateStereoCameras(self, leftCorners, rightCorners, objectPoints):
        """Calibrates the stereo camera."""
        # Prepares the external parameters.
        cameraMatrix = {}
        distCoeffs   = {}
        for index, parameter in zip(range(2), CaptureManager.Instance.Parameters):
            cameraMatrix[index] = parameter.K
            distCoeffs[index]   = parameter.DistCoeffs
        imageSize = CaptureManager.Instance.Size
        
        print cameraMatrix[0]

        # Defines the criterias used by stereoCalibrate() function.
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        flags  = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

        # Calibrates a stereo camera setup.
        (_, cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], R, t, E, F) = cv2.stereoCalibrate(
            objectPoints, leftCorners, rightCorners, cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, criteria=criteria, flags=flags)

        # Records the external parameters.
        for index, parameter in zip(range(2), CaptureManager.Instance.Parameters):
            parameter.K = cameraMatrix[index]
            parameter.DistCoeffs = distCoeffs[index]
            parameter.R = R
            parameter.t = t
            parameter.E = E
            parameter.F = F

        # Return the final result.
        return R, t

    def CrossProductMatrix(self, t):
        """Estimating the skew symmetric matrix."""
        # <008> Estimate the skew symmetric matrix.
        return  [ 
                  [0, -t[2], t[1]],
                  [t[2], 0, -t[0]],
                  [-t[1], t[0], 0]
                ]

    def EssentialMatrix(self, R, t):
        """Calculate the Essential Matrix."""
        # <009> Estimate manually the essential matrix.
        return np.dot(self.CrossProductMatrix(t), R)

    def FundamentalMatrix(self, K1, K2, E):
        """Calculate the Fundamental Matrix."""
        # <010> Estimate manually the fundamental matrix.
        return np.dot(np.linalg.inv(K2).T, np.dot(E, np.linalg.inv(K1)))

    def StereoRectify(self, R, t):
        """Computes rectification transforms for each head of a calibrated stereo camera."""
        # Prepares the external parameters.
        cameraMatrix     = {}
        distCoeffs       = {}
        rotationMatrix   = {}
        projectionMatrix = {}
        for index, parameter in zip(range(2), CaptureManager.Instance.Parameters):
            cameraMatrix[index]     = parameter.K
            distCoeffs[index]       = parameter.DistCoeffs
            rotationMatrix[index]   = None
            projectionMatrix[index] = None
        imageSize = CaptureManager.Instance.Size

        # Computes rectification transforms.
        rotationMatrix[0], rotationMatrix[1], projectionMatrix[0], projectionMatrix[1], Q, _, _ = cv2.stereoRectify(
            cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, t, alpha=0)

        # Records the external parameters.
        for index, parameter in zip(range(2), CaptureManager.Instance.Parameters):
            parameter.R = rotationMatrix[index]
            parameter.P = projectionMatrix[index]
            parameter.Q = Q

    def UndistortRectifyMap(self):
        """Computes the undistortion and rectification transformation maps."""
        # Prepares the external parameters.
        cameraMatrix     = {}
        distCoeffs       = {}
        rotationMatrix   = {}
        projectionMatrix = {}
        maps             = {}
        for parameter in CaptureManager.Instance.Parameters:
            index = parameter.Index
            cameraMatrix[index]     = parameter.K
            distCoeffs[index]       = parameter.DistCoeffs
            rotationMatrix[index]   = parameter.R
            projectionMatrix[index] = parameter.P
        imageSize = CaptureManager.Instance.Size

        # Computes the undistortion and rectification transformation maps
        for parameter in CaptureManager.Instance.Parameters:
            index = parameter.Index
            parameter.Maps = cv2.initUndistortRectifyMap(cameraMatrix[index], distCoeffs[index], rotationMatrix[index], projectionMatrix[index], imageSize, cv2.CV_16SC2)

    def UndistortImages(self, left, right):
        """Method used to undistorte the input stereo images."""
        # Prepares the external parameters.
        maps = {}
        for index, parameter in zip(range(2), CaptureManager.Instance.Parameters):
            maps[index] = parameter.Maps

        # Applies a generic geometrical transformation to each stereo image.
        leftUndistort  = cv2.remap(left,  maps[0][0], maps[0][1], cv2.INTER_LINEAR)
        rightUndistort = cv2.remap(right, maps[1][0], maps[1][1], cv2.INTER_LINEAR)

        # Returns the undistorted images.
        return leftUndistort, rightUndistort

    def StereoSGBM(self, leftImage, rightImage, minDisparity=0, blockSize=1):
        """Computing a stereo correspondence using the block matching algorithm."""
        # Get the disparity-to-depth mapping matrix.
        for parameter in CaptureManager.Instance.Parameters:
            Q = parameter.Q
            break

        # All values used in this function were informed by OpenCV docs.
        sgbm = cv2.StereoSGBM_create(minDisparity, minDisparity + 16, blockSize,
                                     P1=8*3*blockSize**2, P2=32*3*blockSize**2, 
                                     disp12MaxDiff=1, preFilterCap=63,
                                     uniquenessRatio=10, speckleWindowSize=100,
                                     speckleRange=32, mode=cv2.STEREO_SGBM_MODE_HH)

        # Computes disparity map for the specified stereo pair.
        return sgbm.compute(leftImage, rightImage).astype(np.float32) / 16.0, Q

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.Calibration.Calibration object."
