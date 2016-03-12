#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : SIGBTools.py                                             -->
#<!-- Description: Main class of this project                               -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 24/02/2016                                               -->
#<!-- Change     : 24/02/2016 - Creation of this class                      -->
#<!-- Review     : 24/02/2016 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2016022401 $"

########################################################################
import math
import numpy as np
import os
import pickle

from Framework.ImageProcessing.RegionProps        import RegionProps as Props
from Framework.ImageProcessing.ROISelector        import ROISelector as Selector
from Framework.RecordingVideos.RecordingManager   import RecordingManager
from Framework.VideoCaptureDevices.CaptureManager import CaptureManager
from Framework.VideoCaptureDevices.Enumerations   import *

########################################################################

#----------------------------------------------------------------------#
#                         RegionProps Methods                          #
#----------------------------------------------------------------------#
def RegionProps():
    """This class is used for getting descriptors of contour-based connected components.

        The main method to use is: CalcContourProperties(contour, properties=[]):
        contour: a contours found through cv2.findContours()
        properties: list of strings specifying which properties should be calculated and returned

        The following properties can be specified:

        Area: Area within the contour - float
        Boundingbox: Bounding box around contour - 4 tuple (topleft.x, topleft.y, width, height)
        Length: Length of the contour
        Centroid: The center of contour - (x, y)
        Moments: Dictionary of moments
        Perimiter: Permiter of the contour - equivalent to the length
        Equivdiameter: sqrt(4 * Area / pi)
        Extend: Ratio of the area and the area of the bounding box. Expresses how spread out the contour is
        Convexhull: Calculates the convex hull of the contour points
        IsConvex: boolean value specifying if the set of contour points is convex

        Returns: Dictionary with key equal to the property name

        Example: 
             image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
             goodContours = []
             for cnt in contours:
                vals = props.CalcContourProperties(cnt, ["Area", "Length", "Centroid", "Extend", "ConvexHull"])
                if vals["Area"] > 100 and vals["Area"] < 200:
                   goodContours.append(cnt)
    """
    return Props()

#----------------------------------------------------------------------#
#                         ROISelector Methods                          #
#----------------------------------------------------------------------#
def ROISelector(image):
    """This class returns the corners of the selected area as: [(UpperLeftcorner), (LowerRightCorner)].
       Use the Right Mouse Button to set upper left hand corner and the Left Mouse Button to set the lower right corner.
        Click on the image to set the area
        Keys:
            Enter/SPACE - OK
            ESC/q       - Exit (Cancel)"""
    return Selector(image)

#----------------------------------------------------------------------#
#                       RecordingVideos Methods                        #
#----------------------------------------------------------------------#
def RecordingVideos(filepath, size):
    """
    RecordingVideos(filepath, size) -> True or False

    Class for recording videos. The class provides access to OpenCV for recording multiple videos and image sequences.
    Returns: A boolean value inform if the video capture device was correct connected.
    Parameters: filepath: the complete file path of the recorded video,
                size: a tuple with the output video size [Format: (X, Y), where X is the video width and Y is the video height].

    Usage: SIGBTools.RecordingVideos("C:\output.wmv", (640, 480))
    """
    return RecordingManager.Instance.AddVideo(filepath, size=size)

def write(images):
    """
    write(images)

    Writes the next video frame.
    Returns: This method does not return anything.
    Parameters: images: the input images.

    Usage: SIGBTools.write(images)
    """
    return RecordingManager.Instance.Write(images)

def close():
    """
    close()

    Closes video writer.
    Returns: This method does not return anything.
    Parameters: This method does not have any parameter.

    Usage: SIGBTools.close()
    """
    RecordingManager.Instance.Release()

#----------------------------------------------------------------------#
#                         Geometrical Methods                          #
#----------------------------------------------------------------------#
def GetCircleSamples(center=(0, 0), radius=1, numPoints=30):
    """
    GetCircleSamples(center=(0, 0), radius=1, numPoints=30) -> (x, y, d_x, d_y)

    Samples a circle with center center = (x, y), radius = 1 and in numPoints on the circle.
    Returns: an array of a tuple containing the points (x, y) on the circle and the curve gradient in the point (d_x, d_y).
             Notice the gradient (d_x, d_y) has unit length.
    Parameters: center: (x, y) circle center.
                radius: circle radius.
                numPoints: number of points in the circle.

    Usage: P = SIGBTools.GetCircleSamples((100, 100), 40, 20)
    """
    s = np.linspace(0, 2 * math.pi, numPoints)
    P = [ (radius * np.cos(t) + center[0], radius * np.sin(t) + center[1], np.cos(t), np.sin(t)) for t in s ]

    return P

def GetLineCoordinates(p1, p2):
    """
    GetLineCoordinates(p1, p2) -> coordinates

    Get integer coordinates between p1 and p2 using Bresenhams algorithm.
    Returns: a coordinate of I along the line from p1 to p2.
    Parameters: p1: A cartesian coordinate.
                p2: A cartesian coordinate.

    Usage: coordinates = SIGBTools.GetLineCoordinates((x1,y1),(x2,y2))
    """
    (x1, y1) = p1
    x1 = int(x1)
    y1 = int(y1)

    (x2, y2) = p2
    x2 = int(x2)
    y2 = int(y2)

    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True

    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)

    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1

    for x in range(x1, x2 + 1):
        if issteep:
            points.append([y, x])
        else:
            points.append([x, y])
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax

    # Reverse the list if the coordinates were reversed.
    if rev:
        points.reverse()

    retPoints = np.array(points)
    X = retPoints[:, 0]
    Y = retPoints[:, 1]

    return retPoints

#----------------------------------------------------------------------#
#                     VideoCaptureDevices Methods                      #
#----------------------------------------------------------------------#
def VideoCapture(idn=0, enum=CAMERA_VIDEOCAPTURE_640X480_30FPS):
    """
    VideoCapture(idn=0, enum=CAMERA_VIDEOCAPTURE_640X480_30FPS) -> True or False

    Class for video capturing from video files, image sequences or cameras. The class provides access to OpenCV for capturing multiple videos from cameras or for reading video files and image sequences.
    Returns: A boolean value inform if the video capture device was correct connected.
    Parameters: idn: camera index or full path of a image/video file,
                enum: enumeration for a connected camera [Format: SIGBTools.CAMERA_X_Y_Z, where X is the camera model, Y is the camera resolution and Z the number of captured frames per second].

    Usage: SIGBTools.VideoCapture()
           SIGBTools.VideoCapture(0)
           SIGBTools.VideoCapture(1, SIGBTools.CAMERA_PS3EYE_640X480_75FPS)
    """
    return CaptureManager.Instance.AddCamera(idn, enum)

def read():
    """
    read() -> images

    Grabs, decodes and returns the next video frame.
    Returns: A vector of synchronized images.
    Parameters: This method does not have any parameter.

    Usage: images = SIGBTools.read()
    """
    return CaptureManager.Instance.Read()

def release():
    """
    release()

    Closes video file or capturing device.
    Returns: This method does not return anything.
    Parameters: This method does not have any parameter.

    Usage: SIGBTools.release()
    """
    CaptureManager.Instance.Release()

def videoFPS():
    """
    videoFPS() -> fps

    Estimate the current number of captured frames per second.
    Returns: An integer value inform the current framerate.
    Parameters: This method does not have any parameter.

    Usage: fps = SIGBTools.videoFPS()
    """
    return CaptureManager.Instance.FPS

def videoSize():
    """
    videoSize() -> (width, height)

    Get the camera resolution.
    Returns: A map with two integer values, i.e. width and height.
    Parameters: This method does not have any parameter.

    Usage: width, heigh = SIGBTools.videoSize()
           size = SIGBTools.videoSize()
    """
    return CaptureManager.Instance.Size
