#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : Cube.py                                                  -->
#<!-- Description: Class used for creating an augmented cube                -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 18/04/2015                                               -->
#<!-- Change     : 18/04/2015 - Creation of these classes                   -->
#<!--            : 05/12/2015 - Update the class for the new assignment     -->
#<!-- Review     : 05/12/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = '$Revision: 2015051201 $'

########################################################################
import cv2
import numpy as np
import math
import sys

from Framework.VideoCaptureDevices.CaptureManager import CaptureManager
from Framework.ImageProcessing.ImageManager  import ImageManager

########################################################################
class Cube(object):
    """Cube Class is used creating an augmented cube."""

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def Object(self):
        """Get an augmented cube."""
        return self.__Object

    @property
    def CoordinateSystem(self):
        """Get the augmented coordinate system."""
        return self.__CoordinateSystem

    #----------------------------------------------------------------------#
    #                      Augmented Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Augmented Class Constructor."""
        # Creates the augmented objects used by this class.
        self.__CreateObjects()

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def PoseEstimationMethod1(self, image, corners, homographyPoints, calibrationPoints, projectionMatrix, cameraMatrix):
        """This method uses the homography between two views for finding the extrinsic parameters (R|t) of the camera in the current view."""
        
        H1, mask = cv2.findHomography(corners, calibrationPoints)
        
        H1 = np.linalg.inv(H1)
        #H1, mask = cv2.findHomography(homographyPoints, calibrationPoints)
        #H2 = ImageManager.Instance.EstimateHomography(calibrationPoints, corners)
        #H2cs = np.dot(H1, H2)
        ##Normalize homography
        #homography2cs = H2cs / H2cs[2,2]  
        
        P2 = np.dot(H1, projectionMatrix)
        A = np.dot(np.linalg.inv(cameraMatrix), P2[:, :3])
        #print A[:, 0].shape, A[:, 1].shape
        #sys.exit(0)
        A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])]).T
        
        P2[:, :3] = np.dot(cameraMatrix, A)   
        return P2

    def PoseEstimationMethod2(self, corners, patternPoints, cameraMatrix, distCoeffs):
        """This function uses the chessboard pattern for finding the extrinsic parameters (R|T) of the camera in the current view."""
        retval, rvec, tvec = cv2.solvePnP(patternPoints, corners, cameraMatrix, distCoeffs)
        # Convert the rotation vector to a 3 x 3 rotation matrix
        R = cv2.Rodrigues(np.array(rvec))[0]
        # Stack rotationVector and translationVectors so we get a single array
        stacked = np.hstack((R, np.array(tvec)))
        # Now we can get P by getting the dot product of K and the rotation/translation elements
        return np.dot(cameraMatrix, stacked)
    
    def PoseEstimation(self, objectPoints, corners, points, cameraMatrix, distCoeffs):
        """Define the pose estimation of the calibration pattern."""
        # <013> Find the rotation and translation vectors.
        retval, rvec, tvec = cv2.solvePnP(objectPoints, corners, cameraMatrix, distCoeffs)

        # <014> Save the rotation and translation matrices as private attributes.
        self.__rotation = rvec
        self.__translation = tvec
        
        # <015> Project 3D points to image plane.
        imagePoints, jacobian = cv2.projectPoints(points, self.__rotation, self.__translation, cameraMatrix, distCoeffs)

        # Return the final result.
        return imagePoints    

    def DrawCoordinateSystem(self, image):
        """Draw the coordinate axes attached to the chessboard pattern."""
        origin = np.zeros((3 , 1))
        P2 = CaptureManager.Instance.Parameters.P
        origin = CaptureManager.Instance.Parameters.Project(origin)
        
        points = self.CoordinateSystem
        points = CaptureManager.Instance.Parameters.Project(points)
        
        cv2.line(image, tuple(origin[0].astype(int))[:2], tuple(points[0].astype(int))[:2], (255, 0, 0), 3)
        cv2.line(image, tuple(origin[0].astype(int))[:2], tuple(points[1].astype(int))[:2], (0, 255, 0), 3)
        cv2.line(image, tuple(origin[0].astype(int))[:2], tuple(points[2].astype(int))[:2], (0, 0, 255), 3)
        
        cv2.circle(image, tuple(points[0].astype(int))[:2], 2, (255, 0, 0), 12)
        cv2.circle(image, tuple(points[1].astype(int))[:2], 2, (0, 255, 0), 12)
        cv2.circle(image, tuple(points[2].astype(int))[:2], 2, (0, 0, 255), 12)    

    def DrawAugmentedCube(self, image):
        """Draw a cube attached to the chessboard pattern."""
        points = self.Object
        P2 = CaptureManager.Instance.Parameters.P
        points = CaptureManager.Instance.Parameters.Project(points)        
        
        cv2.drawContours(image, [points[:4, :2].astype(int)], -1, (0, 255, 0), cv2.FILLED)

        for i, j in zip(range(4), range(4, 8)):
            cv2.line(image, tuple(points[i, :2].astype(int)), tuple(points[j, :2].astype(int)), (255, 0, 0), 3)

        cv2.drawContours(image, [points[4: , :2].astype(int)], -1, (0, 0, 255), 3)

    def CalculateFaceCornerNormals(self, top, right, left, up, down):
        """Return the normal vector of each cube face."""
        cubeCornerNormals = self.__GetNormalsInCubeCorners(top, right, left, up, down)

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]])
        t = cubeCornerNormals[i, j].T

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[6, 2, 3, 7], [6, 2, 3, 7], [6, 2, 3, 7]])
        r = cubeCornerNormals[i, j].T

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[4, 0, 1, 5], [4, 0, 1, 5], [4, 0, 1, 5]])
        l = cubeCornerNormals[i, j].T

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[5, 1, 2, 6], [5, 1, 2, 6], [5, 1, 2, 6]])
        u = cubeCornerNormals[i, j].T

        i = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        j = np.array([[7, 3, 0, 4], [7, 3, 0, 4], [7, 3, 0, 4]])
        d = cubeCornerNormals[i, j].T

        return t, r, l, u, d

    def GetFaceNormal(self, points):
        """Get some information of a correspoding cube face."""
        # Estimate the normal vector of the corresponding cube face.
        A = np.subtract([points[1, 0], points[1, 1], points[1, 2]], [points[0, 0], points[0, 1], points[0, 2]])
        B = np.subtract([points[2, 0], points[2, 1], points[2, 2]], [points[0, 0], points[0, 1], points[0, 2]])
        normal = np.cross(A, B)
        normal = normal / np.linalg.norm(normal)

        # Calculate the midpoint of the corresponding cube face.
        center = np.mean(points, axis=0)

        # Estimate the vector from camera center to cube face center.
        cameraCenter = -np.dot(self.__rotation.T, self.__translation).T
        C = np.subtract(cameraCenter, center)
        C = C / np.linalg.norm(C)

        # Calculate the angle of the normal vector.
        angle = np.arccos(np.dot(C, normal)) * (180 / np.pi)

        # Return the results.
        return normal, center, angle

    #----------------------------------------------------------------------#
    #                         Private Class Methods                        #
    #----------------------------------------------------------------------#
    def __CreateObjects(self):
        """Defines the points of the augmented objects based on calibration patterns."""
        # Creates an augmented cube.
        self.__Object = np.float32([[3, 1,  0], [3, 4,  0], [6, 4,  0], [6, 1,  0],
                                    [3, 1, -3], [3, 4, -3], [6, 4, -3], [6, 1, -3]]).T
        # Creates the coordinate system.
        self.__CoordinateSystem = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, -2]]).reshape(-1, 3)

    def __GetNormalsInCubeCorners(self, top, right, left, up, down):
        """Estimate the normal face through the cube corners."""
        points = []

        points.append((self.GetFaceNormal(up)[0]   + self.GetFaceNormal(left)[0]  - self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(up)[0]   + self.GetFaceNormal(right)[0] - self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(down)[0] + self.GetFaceNormal(right)[0] - self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(down)[0] + self.GetFaceNormal(left)[0]  - self.GetFaceNormal(top)[0]) / 3)

        points.append((self.GetFaceNormal(up)[0]   + self.GetFaceNormal(left)[0]  + self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(up)[0]   + self.GetFaceNormal(right)[0] + self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(down)[0] + self.GetFaceNormal(right)[0] + self.GetFaceNormal(top)[0]) / 3)
        points.append((self.GetFaceNormal(down)[0] + self.GetFaceNormal(left)[0]  + self.GetFaceNormal(top)[0]) / 3)

        return np.array(points).T

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Get a object representation in a string format."""
        return "Framework.Augumented.Cube object."
