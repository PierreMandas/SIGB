#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : Assignment3.py                                           -->
#<!-- Description: Main class of Assignment #03                             -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 24/02/2016                                               -->
#<!-- Change     : 24/02/2016 - Creation of this class                      -->
#<!-- Review     : 26/04/2016 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2016042601 $"

########################################################################
import cv2
import os
import numpy as np
import warnings

from collections import deque

from pylab import draw
from pylab import figure
from pylab import plot
from pylab import show
from pylab import subplot
from pylab import title

import SIGBTools

########################################################################
class Assignment3(object):
    """Assignment3 class is the main class of Assignment #03."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __path = "./Assignments/_03/"

    __plyHeader = '''ply
format ascii 1.0
element vertex %(num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

    #----------------------------------------------------------------------#
    #                    Assignment3 Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Assignment3 Class Constructor."""
        warnings.simplefilter("ignore")
        self.__Clear()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Start(self):
        """Start Assignment #03."""
        option = "-1"
        clear  = lambda: os.system("cls" if os.name == "nt" else "clear")
        while option != "0":
            clear()
            print "\n\t#################################################################"
            print "\t#                                                               #"
            print "\t#                         Assignment #03                        #"
            print "\t#                                                               #"
            print "\t#################################################################\n"
            print "\t[1] Fundamental Matrix."
            print "\t[2] Stereo Camera Calibration."
            print "\t[3] Texture Mapping."
            print "\t[4] Principal Component Analysis (PCA)."
            print "\t[0] Back.\n"
            option = raw_input("\tSelect an option: ")

            if option == "1":
                self.__EpipolarGeometry()
            elif option == "2":
                self.__StereoCamera()
            elif option == "3":
                self.__TextureMapping()
            elif option == "4":
                self.__PCA()
            elif option != "0":
                raw_input("\n\tINVALID OPTION!!!")

    #----------------------------------------------------------------------#
    #                        Private Class Methods                         #
    #----------------------------------------------------------------------#
    def __EpipolarGeometry(self):
        """Define the epipolar geometry between stereo cameras."""
        # Creates a window to show the stereo images.
        cv2.namedWindow("Stereo",  cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Stereo", self.__FMEyeMouseEvent)

        # Load two video capture devices.
        SIGBTools.VideoCapture(0, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)
        SIGBTools.VideoCapture(1, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)

        # Repetition statement for analyzing each captured image.
        while True:
            # Check if the fundamental matrix process is running.
            if not self.__isFrozen:
                # Grab the video frames.
                leftImage, rightImage = SIGBTools.read()
                # Combine two stereo images in only one window.
                self.__Image = self.__CombineImages(leftImage, rightImage, 0.5)

            # Check what the user wants to do.
            inputKey = cv2.waitKey(1)
            # Esc or letter "q" key.
            if inputKey == 27 or inputKey == ord("q"):
                break
            # Letter "f" key.
            elif inputKey == ord("f"):
                self.__isFrozen = not self.__isFrozen

            # Show the final processed image.
            cv2.imshow("Stereo", self.__Image)

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()

    def __StereoCamera(self):
        """Define the epipolar geometry between stereo cameras."""
        # Load two video capture devices.
        SIGBTools.VideoCapture(0, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)
        SIGBTools.VideoCapture(1, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)

        # Calibrate each individual camera.
        SIGBTools.calibrate()

        # Creates a window to show the stereo images.
        cv2.namedWindow("Stereo",  cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Stereo", self.__SCEyeMouseEvent)
        self.__Disparity = np.zeros((1, 1, 1))

        # Repetition statement for analyzing each captured image.
        while True:
            # Grab the video frames.
            leftImage, rightImage = SIGBTools.read()

            # Find the pattern in the image.
            leftCorners  = SIGBTools.FindCorners(leftImage)
            rightCorners = SIGBTools.FindCorners(rightImage)

            # Check if the calibration process is running.
            if self.__isCalibrating:
                # If both pattern have been recognized, start the calibration process.
                if leftCorners is not None and rightCorners is not None:
                    self.__Calibrate(leftCorners, rightCorners)
                # Otherwise, stop the calibrations process.
                else:
                    self.__isCalibrating = False

            # Combine two stereo images in only one window.
            self.__Image = self.__CombineImages(leftImage, rightImage, 0.5)

            # Undistort the stereo images.
            if self.__isUndistort:
                leftUndistort, rightUndistort = SIGBTools.UndistortImages(leftImage, rightImage)
                self.__Undistort = self.__CombineImages(leftUndistort, rightUndistort, 0.5)
                if self.__isDepth:
                    self.__Disparity = self.__DepthMap(leftUndistort, rightUndistort)

            # Check what the user wants to do.
            inputKey = cv2.waitKey(1)
            # Esc or letter "q" key.
            if inputKey == 27 or inputKey == ord("q"):
                break
            # Space key.
            elif inputKey == 32:
                self.__isCalibrating = True
            # Letter "s" key.
            elif inputKey == ord("s") and self.__isDepth:
                self.__isSaving = True
            elif inputKey == ord("d"):
                if not self.__isDepth:
                    # Creates a window to show the depth map.
                    cv2.namedWindow("DepthMap", cv2.WINDOW_AUTOSIZE)
                    cv2.createTrackbar("minDisparity", "DepthMap", 1, 32, self.__SetMinDisparity)
                    cv2.createTrackbar("blockSize",    "DepthMap", 1,  5, self.__SetNothing)
                    self.__isDepth = True
                else:
                    cv2.destroyWindow("DepthMap")
                    self.__isDepth = False

            # Show the final processed image.
            cv2.imshow("Stereo", self.__Image)
            if self.__isUndistort:
                cv2.imshow("Undistort", self.__Undistort)
                if self.__isDepth:
                    cv2.imshow("DepthMap", self.__Disparity)

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()

    def __TextureMapping(self):
        """Apply a texture mapping on an augmented object."""
        # Creates a window to show the stereo images.
        cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)

        # Load two video capture devices.
        SIGBTools.VideoCapture(0, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)

        # Repetition statement for analyzing each captured image.
        while True:
            # Grab the video frames.
            image = SIGBTools.read()

            # Find the pattern in the image.
            corners = SIGBTools.FindCorners(image, False)

            # Apply the augmented object.
            if corners is not None:
                image = self.__Augmentation(corners, image)

            # Check what the user wants to do.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Show the final processed image.
            cv2.imshow("Original", image)

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()

    def __PCA(self):
        """Principal Component Analysis."""
        pass

    #----------------------------------------------------------------------#
    #           Private Class Methods Developed by the Students            #
    #----------------------------------------------------------------------#
    def __FundamentalMatrix(self, point):
        # Check if the image is frozen.
        # SIGB: The user can frozen the input image presses "f" key.
        if self.__isFrozen:

            # Insert the new selected point in the queue.
            if self.__UpdateQueue(point):

                # Get all points selected by the user.
                points = np.asarray(self.PointsQueue, dtype=np.float32)
                points = np.array([np.array([x/2, y/2, z]) for x, y, z in [point for point in points]])
                
                # <000> Get the selected points from the left and right images.
                leftPoints = points[::2]
                rightPoints = np.array([[x+320, y, z] for x, y, z in points[1::2]])
                
                #print leftPoints
                #print rightPoints

                # <001> Estimate the Fundamental Matrix.
                F, mask = cv2.findFundamentalMat(leftPoints, rightPoints)

                # Get each point from left image.
                # for point in leftPoints:
                for point in leftPoints:
                    #pass
                    # <002> Estimate the epipolar line.
                    rightEpiLine = np.dot(F, point)

                    # <003> Define the initial and final points of the line.
                    a, b, c = rightEpiLine
                    x0 = 320
                    x1 = 640
                    y0 = int(-(c+a*x0)/b)
                    y1 = int(-(c+a*x1)/b)
                    
                    # <004> Draws the epipolar line in the input image.
                    cv2.line(self.__Image, (x0, y0), (x1, y1), (255, 0, 0), 1)

                # Get each point from right image.
                # for point in rightPoints:
                for point in rightPoints:
                    #pass
                    # <005> Estimate the epipolar line.
                    
                    leftEpiLine = np.dot(F.transpose(), point)
                    
                    a, b, c = leftEpiLine
                    x0 = 0
                    x1 = 320
                    y0 = int(-(c+a*x0)/b)
                    y1 = int(-(c+a*x1)/b)
                    
                    cv2.line(self.__Image, (x0, y0), (x1, y1), (0, 0, 255), 1)

    def __Calibrate(self, leftCorners, rightCorners):
        """Calibrate the stereo camera for each new detected pattern."""
        # Get The outer vector contains as many elements as the number of the pattern views.
        objectPoints = SIGBTools.CalculatePattern()

        # <006> Insert the pattern detection results in three vectors.
        self.__LeftCorners.append(leftCorners)
        self.__RightCorners.append(rightCorners)
        self.__ObjectPoints.append(objectPoints)

        # <007> Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
        leftCameraMatrix, leftDistCoeffs = SIGBTools.CalibrationManager.Instance.CalibrateCamera(0, self.__LeftCorners, self.__ObjectPoints, (640,480))
        rightCameraMatrix, rightDistCoeffs = SIGBTools.CalibrationManager.Instance.CalibrateCamera(1, self.__RightCorners, self.__ObjectPoints, (640,480))
        
        # Calibrates the stereo camera.
        R, t = SIGBTools.calibrateStereoCameras(self.__LeftCorners, self.__RightCorners, self.__ObjectPoints)

        # <011> Computes rectification transforms for each head of a calibrated stereo camera.
        SIGBTools.StereoRectify(R, t)

        # <012> Computes the undistortion and rectification transformation maps.
        SIGBTools.UndistortRectifyMap()

        # End the calibration process.
        self.__isCalibrating = False
        self.__isUndistort = True

        # Stop the system for 1 second, because the user will see the processed images.
        cv2.waitKey(1000)

    def __Augmentation(self, corners, image):
        """Draws some augmentated object in the input image."""
        # Defines the result image.
        result = image.copy()

        # Get The outer vector contains as many elements as the number of the pattern views.
        objectPoints = SIGBTools.CalculatePattern()

        # Prepares the external parameters.
        P, K, R, t, distCoeffs = SIGBTools.GetCameraParameters()

        # Defines the pose estimation of the coordinate system.
        P2 = SIGBTools.PoseEstimationMethod2(corners, objectPoints, K, distCoeffs)
        SIGBTools.SetCameraParameters(P2, K, R, t, distCoeffs)

        # Draws the coordinate system over the chessboard pattern.
        SIGBTools.DrawCoordinateSystem(result)

        # Get the points of the cube.
        SIGBTools.DrawAugmentedCube(result)

        # Threshould used for selecting which cube faces will be drawn.
        threshold = 88

        # Define each correspoding cube face.
        cube = np.float32([[3, 1,  0], [3, 4,  0], [6, 4,  0], [6, 1,  0],
                           [3, 1, -3], [3, 4, -3], [6, 4, -3], [6, 1, -3]])
        
        
        faces = [
                    np.array([cube[4], cube[5], cube[6], cube[7]]),
                    np.array([cube[5], cube[2], cube[6], cube[1]]),
                    np.array([cube[7], cube[3], cube[0], cube[4]]),
                    np.array([cube[4], cube[0], cube[1], cube[5]]),
                    np.array([cube[6], cube[2], cube[3], cube[7]])
                ]         
        
        
        # Estimates the cube coordinates on the image.
        cube = SIGBTools.PoseEstimation(objectPoints, corners, cube, K, distCoeffs)
        cube = cube.reshape(8, -1)
        
        topFace   = cube[4:]
        upFace    = np.vstack([cube[5], cube[1:3], cube[6]])
        downFace  = np.vstack([cube[7], cube[3],   cube[0], cube[4]])
        leftFace  = np.vstack([cube[4], cube[0:2], cube[5]])
        rightFace = np.vstack([cube[6], cube[2:4], cube[7]])         

        # <020> Applies the texture mapping over all cube sides.
        # Method to get the x and y coordinate tuple of a 1x2 array
        def mkPoint(_2dPointArray):
            return (int(_2dPointArray[0]), int(_2dPointArray[1]))        
        
        # Get all faceNormals. Get the center of each face (startpoint of face normal). Calculate the face normal vector of each face (endpoint of face normal).
        faceNormals = [SIGBTools.GetFaceNormal(face) for face in faces]
        faceCenters = np.array([center for _, center, _ in faceNormals])
        faceNormalvectors = np.array([np.add(center, normal) for normal, center, _ in faceNormals])
        
        # Project the face centers and the face normal vectors.
        faceCentersProjected = SIGBTools.PoseEstimation(objectPoints, corners, faceCenters, K, distCoeffs)
        faceNormalvectorsProjected = SIGBTools.PoseEstimation(objectPoints, corners, faceNormalvectors, K, distCoeffs)
        
        # Apply texture to each face of the cube and draw the normals, if the angle is below the threshold value of 90.
        if faceNormals[0][2] < threshold:
            self.__ApplyTexture(result, 'Assignments/_03/Images/Top.jpg', topFace)
            cv2.line(result, mkPoint(faceCentersProjected[0][0]), mkPoint(faceNormalvectorsProjected[0][0]), (255, 0, 0), thickness=5)
            
        if faceNormals[1][2] < threshold:
            self.__ApplyTexture(result, 'Assignments/_03/Images/Up.jpg', upFace)         
            cv2.line(result, mkPoint(faceCentersProjected[1][0]), mkPoint(faceNormalvectorsProjected[1][0]), (255, 0, 0), thickness=5)
        
        if faceNormals[2][2] < threshold:
            self.__ApplyTexture(result, 'Assignments/_03/Images/Down.jpg', downFace)
            cv2.line(result, mkPoint(faceCentersProjected[2][0]), mkPoint(faceNormalvectorsProjected[2][0]), (255, 0, 0), thickness=5)
        
        if faceNormals[3][2] < threshold:
            self.__ApplyTexture(result, 'Assignments/_03/Images/Left.jpg', leftFace)
            cv2.line(result, mkPoint(faceCentersProjected[3][0]), mkPoint(faceNormalvectorsProjected[3][0]), (255, 0, 0), thickness=5)
        
        if faceNormals[4][2] < threshold:
            self.__ApplyTexture(result, 'Assignments/_03/Images/Right.jpg', rightFace)
            cv2.line(result, mkPoint(faceCentersProjected[4][0]), mkPoint(faceNormalvectorsProjected[4][0]), (255, 0, 0), thickness=5)

        # Return the result image.
        return result

    def __ApplyTexture(self, image, filename, points):
        """Applies a texture mapping over an augmented virtual object."""
        # Get the size of the analyzed image.
        h, w = image.shape[:2]

        # <016> Open the texture mapping image and get its size.
        texture = cv2.imread(filename)
        texH, texW = texture.shape[:2]

        # Creates a mask with the same size of the input image.
        whiteMask = np.ones(texture.shape, dtype=np.uint8) * 255

        # <017> Estimate the homography matrix between the texture mapping and the cube face.
        srcPoints = np.array([
                                (0, 0),
                                (0, texH),
                                (texW, 0),
                                (texW, texH)
                             ])
        
        dstPoints = np.array([            
                                points[0],
                                points[1],
                                points[3],
                                points[2]
                             ])
        homography, _ = cv2.findHomography(srcPoints, dstPoints)

        # <018> Applies a perspective transformation to the texture mapping image.
        texture = cv2.warpPerspective(texture, homography, (w, h))
        whiteMask = cv2.warpPerspective(whiteMask, homography, (w, h))

        # <019> Create a mask from the cube face using the texture mapping image.
        whiteMask = np.array(cv2.bitwise_or(texture, whiteMask), np.uint8)
        mapping = cv2.bitwise_not(whiteMask, image.copy())
        mapping = cv2.bitwise_and(mapping, image.copy())
        cv2.add(texture, mapping, image)

    #----------------------------------------------------------------------#
    #             Private Class Methods Used by Assignment #03             #
    #----------------------------------------------------------------------#
    def __UpdateQueue(self, point):
        """Insert a new point in the queue."""
        # Get the current queue size.
        size = len(self.PointsQueue)

        # Check if the queue is full.
        if size == self.PointsQueue.maxlen:
            return True

        # Defines the color used for draw the circle and the line.
        color = (0, 0, 255) if size % 2 == 0 else (255, 0, 0)

        # Draw a circle in the selected point.
        cv2.circle(self.__Image, point, 3, color, thickness=-1)

        # Adjust the right click to correct position.
        if size % 2 != 0:
            point = (point[0] - 320, point[1])

        # It is necessary to update the selected point, because the systems shows a resized input image.
        # SIBG: You can use the original size, if you call __CombineImages() method with scale factor value 1.0.
        point = (point[0] * 2, point[1] * 2, 1)

        # Insert the new point in the queue.
        self.PointsQueue.append(point)

        # Check if the queue is full now.
        if size + 1 == self.PointsQueue.maxlen:
            return True

        # It is necessary to add more points.
        return False

    def __DepthMap(self, leftImage, rightImage):
        """Estimate the depth map from two stereo images."""
        # Get the attributes for the block matching algorithm.
        # minDisparity needs to be divisible by 16 and block size needs to be an odd number.
        minDisparity  = cv2.getTrackbarPos("minDisparity", "DepthMap")
        minDisparity *= 16
        blockSize = cv2.getTrackbarPos("blockSize", "DepthMap")
        blockSize = 2 * blockSize + 1

        # Computing a stereo correspondence using the block matching algorithm.
        disparity, Q = SIGBTools.StereoSGBM(leftImage, rightImage, minDisparity, blockSize)

        # Check if it is necessary to save the PLY file.
        if self.__isSaving:
            self.__SavePLY(disparity, leftImage, Q)

        # Normalizes the disparity image for a valid output OpenCV image.
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Define the new size to input images.
        disparity = cv2.resize(disparity, (320, 240))

        # Return the depth map image.
        return disparity

    def __SavePLY(self, disparity, image, Q):
        """Save the depth map into a PLY file."""
        # Reprojects a disparity image to 3D space.
        points = cv2.reprojectImageTo3D(disparity, Q)
        colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Creates a mask of the depth mapping matrix.
        mask = disparity > disparity.min()
        points = points[mask].reshape(-1, 3)
        colors = colors[mask].reshape(-1, 3)

        # Defines the output numpy array.
        output = np.hstack([points, colors])

        # Save the output file.
        with open(self.__path + "Outputs/Assignment3.ply", "w") as filename:
            filename.write(self.__plyHeader % dict(num = len(output)))
            np.savetxt(filename, output, "%f %f %f %d %d %d", newline="\n")            

        # End the PLY save process.
        self.__isSaving = False

    def __CombineImages(self, image1, image2, scale=1):
        """Combine two image in only one visualization."""
        # Define the final size.
        height, width = image1.shape[:2]
        width  = int(width  * scale)
        height = int(height * scale)

        # Define the new size to input images.
        image1 = cv2.resize(image1, (width, height))
        image2 = cv2.resize(image2, (width, height))

        # Create the final image.
        image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        image[:height,      :width    ]  = image1
        image[:height, width:width * 2] = image2

        # Return the combine images.
        return image

    def __SetMinDisparity(self, value):
        """Masks the minDisparity variable."""
        if value == 0:
            cv2.setTrackbarPos("minDisparity", "DepthMap", int(1))

    def __SetNothing(self, value):
        """Standard mask."""
        pass

    def __Clear(self):
        """Empty all internal parameters used for this class."""
        self.__isCalibrating = self.__isSaving = self.__isFrozen = self.__isUndistort = self.__isDepth = False
        self.PointsQueue = deque(maxlen=16)
        self.__F = np.zeros((3, 3), dtype=np.float64)
        self.__LeftCorners  = []
        self.__RightCorners = []
        self.__ObjectPoints = []

    #----------------------------------------------------------------------#
    #               Class Calibration Action Events Methods                #
    #----------------------------------------------------------------------#
    def __FMEyeMouseEvent(self, event, x, y, flag, param):
        """This is an example of a calibration process using the mouse clicks."""
        # Check if the system is frozen.
        if not self.__isFrozen:
            return

        # Insert a new point in the calibration process.
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__FundamentalMatrix((x, y))

        # Reset all configuration variables.
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.__Clear()

    def __SCEyeMouseEvent(self, event, x, y, flag, param):
        """This is an example of a calibration process using the mouse clicks."""
        # Reset all configuration variables.
        if event == cv2.EVENT_MBUTTONDOWN:
            self.__Clear()

        # Starts the calibration process.
        elif event == cv2.EVENT_RBUTTONUP:
            self.__isCalibrating = True
