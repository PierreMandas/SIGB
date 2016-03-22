#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : Assignment1.py                                           -->
#<!-- Description: Main class of Assignment #01                             -->
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
import SIGBTools

import cv2
import os
import sys
import numpy as np
import scipy as sp
import time
import warnings
import math as m

from pylab import draw
from pylab import figure
from pylab import imshow
from pylab import plot
from pylab import show
from pylab import subplot
from pylab import title
from pylab import cm, quiver

from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq

########################################################################
class Assignment1(object):
    """Assignment1 class is the main class of Assignment #01."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __path = "./Assignments/_01/"

    #----------------------------------------------------------------------#
    #                           Class Properties                           #
    #----------------------------------------------------------------------#
    @property
    def OriginalImage(self):
        """Get the current original image."""
        return self.__OriginalImage

    @OriginalImage.setter
    def OriginalImage(self, value):
        """Set the current original image."""
        self.__OriginalImage = value

    @property
    def ResultImage(self):
        """Get the result image."""
        return self.__ResultImage

    @ResultImage.setter
    def ResultImage(self, value):
        """Set the result image."""
        self.__ResultImage = value

    @property
    def LeftTemplate(self):
        """Get the left eye corner template."""
        return self.__LeftTemplate

    @LeftTemplate.setter
    def LeftTemplate(self, value):
        """Set the left eye corner template."""
        self.__LeftTemplate = value

    @property
    def RightTemplate(self):
        """Get the right eye corner template."""
        return self.__RightTemplate

    @RightTemplate.setter
    def RightTemplate(self, value):
        """Set the right eye corner template."""
        self.__RightTemplate = value

    @property
    def FrameNumber(self):
        """Get the current frame number."""
        return self.__FrameNumber

    @FrameNumber.setter
    def FrameNumber(self, value):
        """Set the current frame number."""
        self.__FrameNumber = value
        
    @property
    def Figure(self):
        return self.__Figure
    
    @Figure.setter
    def Figure(self, value):
        self.__Figure = value

    #----------------------------------------------------------------------#
    #                    Assignment1 Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Assignment1 Class Constructor."""
        warnings.simplefilter("ignore")
        self.OriginalImage = np.ones((1, 1, 1), dtype=np.uint8)
        self.ResultImage   = np.ones((1, 1, 1), dtype=np.uint8)
        self.LeftTemplate  = []
        self.RightTemplate = []
        self.FrameNumber   = 0
        self.Figure        = False

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Start(self):
        """Start the eye tracking system."""
        # Show the menu.
        self.__Menu()
        # Read a video file.
        filename = raw_input("\n\tType a filename from \"Inputs\" folder: ")
        #filename = "eye01.avi"
        filepath = self.__path + "/Inputs/" + filename
        if not os.path.isfile(filepath):
            print "\tInvalid filename!"
            time.sleep(1)
            return

        # Show the menu.
        self.__Menu()

        # Define windows for displaying the results and create trackbars.
        self.__SetupWindowSliders()

        # Load the video file.
        SIGBTools.VideoCapture(filepath)

        # Shows the first frame.
        self.OriginalImage = SIGBTools.read()
        self.FrameNumber = 1
        self.__UpdateImage()

        # Initial variables.
        saveFrames = False

        # Read each frame from input video.
        while True:            
            # Extract the values of the sliders.
            sliderVals = self.__GetSliderValues()

            # Read the keyboard selection.
            ch = cv2.waitKey(1)

            # Select regions in the input images.
            if ch is ord("m"):
                if not sliderVals["Running"]:
                    roiSelector = SIGBTools.ROISelector(self.OriginalImage)
                    
                    points, regionSelected = roiSelector.SelectArea("Select eye corner", (400, 200))
                    if regionSelected:
                        self.LeftTemplate = self.OriginalImage[points[0][1]:points[1][1],points[0][0]:points[1][0]]
                            
                    points, regionSelected = roiSelector.SelectArea("Select eye corner", (400, 200))
                    if regionSelected:
                        self.RightTemplate = self.OriginalImage[points[0][1]:points[1][1],points[0][0]:points[1][0]]
                    
            # Recording a video file.
            elif ch is ord("s"):
                if saveFrames:
                    SIGBTools.close()
                    saveFrames = False
                else:
                    resultFile = raw_input("\n\tType a filename (result.wmv): ")
                    resultFile = self.__path + "/Outputs/" + resultFile
                    if os.path.isfile(resultFile):
                        print "\tThis file exist! Try again."
                        time.sleep(1)
                        self.__Menu()
                        continue
                    elif not resultFile.endswith("wmv"):
                        print "\tThis format is not supported! Try again."
                        time.sleep(1)
                        self.__Menu()
                        continue
                    self.__Menu()
                    size = self.OriginalImage.shape
                    SIGBTools.RecordingVideos(resultFile, (size[1], size[0]))
                    saveFrames = True

            # Spacebar to stop or start the video.
            elif ch is 32:
                cv2.setTrackbarPos("Stop/Start", "TrackBars", not sliderVals["Running"])

            # Restart the video.
            elif ch is ord("r"):
                # Release all connected videos/cameras.
                SIGBTools. release()
                time.sleep(0.5)

                # Load the video file.
                SIGBTools.VideoCapture(filepath)

                # Shows the first frame.
                self.OriginalImage = SIGBTools.read()
                self.FrameNumber = 1
                self.__UpdateImage()

            # Quit the eye tracking system.
            elif ch is 27 or ch is ord("q"):
                break

            # Check if the video is running.
            if sliderVals["Running"]:
                self.OriginalImage = SIGBTools.read()
                self.FrameNumber += 1
                self.__UpdateImage()

                if saveFrames:
                    SIGBTools.write(self.ResultImage)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()
        if saveFrames:
            SIGBTools.close()

    #----------------------------------------------------------------------#
    #                        Private Class Methods                         #
    #----------------------------------------------------------------------#
    def __Menu(self):
        """Menu of Assignment #01."""
        clear  = lambda: os.system("cls" if os.name == "nt" else "clear")
        clear()
        print "\n\t#################################################################"
        print "\t#                                                               #"
        print "\t#                         Assignment #01                        #"
        print "\t#                                                               #"
        print "\t#################################################################\n"
        print "\t[R] Reload Video."
        print "\t[M] Mark the Region of Interested (RoI) when the video has paused."
        print "\t[S] Toggle video writing."
        print "\t[SPACE] Pause."
        print "\t[Q] or [ESC] Stop and Back.\n"

    #----------------------------------------------------------------------#
    #                         Eye Features Methods                         #
    #----------------------------------------------------------------------#
    def __GetPupil(self, grayscale, threshold, minSize, maxSize, minExtend, maxExtend):
        """Given a grayscale level image and a threshold value returns a list of pupil candidates."""

        # Create a binary image.
        val, threshold = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphology by using an ellipse kernel with a closing and opening of 1 iteration each.
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)) # Elipse-shaped kernel
        #threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=1)
        #threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Show thresholded image
        cv2.imshow("Pupil threshold", threshold)

        # Get the blob properties.
        props = SIGBTools.RegionProps()

        # Calculate blobs.
        _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pupils = []

        # Iterate through each blob and calculate properties of the blob
        for cnt in contours:
            p = props.CalcContourProperties(cnt, ["centroid", "area", "extend"])
            #x,y = p["Centroid"]
            area = p["Area"]
            extend = p["Extend"]
            
            if area > minSize and area < maxSize and extend > minExtend and extend < maxExtend:
                pupilEllipse = cv2.fitEllipse(cnt)
                pupils.append(pupilEllipse)
                #cv2.circle(self.ResultImage, (int(x),int(y)), 2, (0,0,255), 4)
                
        return pupils

    def __DetectPupilKMeans(self, grayscale, K=2, distanceWeight=2, reSize=(40, 40)):
        """Detects the pupil in the grayscale image using k-means.
           grayscale         : input grayscale image
           K                 : Number of clusters
           distanceWeight    : Defines the weight of the position parameters
           reSize            : the size of the image to do k-means on
        """
        # Resize for faster performance and use gaussianBlur to remove noise.
        smallI = cv2.resize(grayscale, reSize)
        smallI = cv2.GaussianBlur(smallI, (3,3), 20)
        M, N = smallI.shape

        # Generate coordinates in a matrix.
        X, Y = np.meshgrid(range(M), range(N))
        
        # Make coordinates and intensity into one vectors.
        z = smallI.flatten()
        x = X.flatten()
        y = Y.flatten()
        O = len(x)  

        # Make a feature vectors containing (x, y, intensity).
        features = np.zeros((O, 3))
        features[:, 0] = z
        features[:, 1] = y / distanceWeight # Divide so that the distance of position weighs less than intensity.
        features[:, 2] = x / distanceWeight
        features = np.array(features, "f")

        # Cluster data.
        centroids, variance = kmeans(features, K)

        # Use the found clusters to map.
        label, distance = vq(features, centroids)

        # Re-create image from.
        labelIm = np.array(np.reshape(label,(M,N)))
			
        f = figure("K-Means")
        imshow(labelIm)
        f.canvas.draw()
        f.show()
        return labelIm, centroids
    
    def __PupilFromKMeans(self, labelIm, centroids, grayscale, minSize, maxSize, minExtend, maxExtend):
        # Binary image created from labelIm
        binLabelIm = np.zeros(labelIm.shape, dtype='uint8')
        
         # Label index from cluster with lowest intensity
        label = np.argmin(centroids[:,0])
        
        # Change every pixel intensity of cluster pixels to 255
        binLabelIm[labelIm == label] = 255
        
        # Resize image to original size
        w, h = grayscale.shape
        binLabelIm = cv2.resize(binLabelIm, (h, w))
        
        # Show the binary label image
        cv2.imshow("Pupil Threshold", binLabelIm)
        
        # Calculate blobs
        _, contours, hierachy = cv2.findContours(binLabelIm, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        pupils = []
        props = SIGBTools.RegionProps()
        
        for cnt in contours:
            p = props.CalcContourProperties(cnt, ["centroid", "area", "extend"])
            x,y = p["Centroid"]
            area = p["Area"]
            extend = p["Extend"]
                        
            if area > minSize and area < maxSize and extend > minExtend and extend < maxExtend:
                pupilEllipse = cv2.fitEllipse(cnt)
                pupils.append(pupilEllipse)
                                 
        return pupils
    
    def __ThresholdFromKMeans(self, centroids):
        threshold = 255
        for c in centroids:
            if c[0] < threshold:
                threshold = c[0]
        
        return threshold
    

    def __DetectPupilHough(self, grayscale):
        """Performs a circular hough transform in the grayscale image and shows the detected circles.
           The circle with most votes is shown in red and the rest in green colors."""
        # See help for http://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html?highlight=hough#cv2.HoughCircles
        blur = cv2.GaussianBlur(grayscale, (31, 31), 11)

        dp = 6 # Inverse ratio of the accumulator resolution to the image resolution.
        minDist = 30 # Minimum distance between the centers of the detected circles.
        highThr = 20 # High threshold for canny.
        accThr = 850 # Accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.
        minRadius = 50 # Minimum circle radius.
        maxRadius = 155 # Maximum circle radius.

        # Apply the hough transform.
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp, minDist, None, highThr, accThr, minRadius, maxRadius)

        # Make a color image from grayscale for display purposes.
        results = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

        # Print all detected circles
        if circles is not None:
            # Print all circles.
            all_circles = circles[0]
            M, N = all_circles.shape
            k = 1
            for circle in all_circles:
                cv2.circle(results, tuple(circle[0:2]), circle[2], (int(k * 255 / M), k * 128, 0))
                circle = all_circles[0,:]
                cv2.circle(results, tuple(circle[0:2]), circle[2], (0, 0, 255), 5)
                k = k + 1

        # Return the result image.
        return results

    def __GetGlints(self, grayscale, threshold, minSize, maxSize):
        """Given a grayscale level image and a threshold value returns a list of glint candidates."""
        glints = []
        
        # Create a binary image.
        val, threshold = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)
        
        # Create kernel and use closing with 2 iterations, with the kernel applied
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
        
        cv2.imshow("Glint threshold", threshold)
        
        # Get the blob properties.
        props = SIGBTools.RegionProps()
        
        # Calculate blobs.
        _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterate through blob and calculate properties of each blob.
        for cnt in contours:
            p = props.CalcContourProperties(cnt, ["centroid", "area"])
            x,y = p["Centroid"]
            area = p["Area"]
            
            if area > minSize and area < maxSize:
                glints.append((x,y))
                
        return glints

    def __FilterPupilGlint(self, pupils, glints):
        """Given a list of pupil candidates and glint candidates returns a list of pupil and glints."""
        filteredPupils = pupils
        filteredGlints = []
        matchedPupils = []
        matchedGlints = []       
        
        sliderVals = self.__GetSliderValues()
        
        for currentGlt in range(len(glints) - 1):
            for nextGlt in range(currentGlt + 1, len(glints)):
        
                glint1 = glints[currentGlt][0], glints[currentGlt][1]
                glint2 = glints[nextGlt][0], glints[nextGlt][1]
        
                distance = self.__EuclideanDistance(glint1, glint2)
        
                # We can use this find out what values to filter between
                #print distance
        
                # Our glints seem to have a distance of 49.xxxx
                # Lets set the filter between this value
                if sliderVals['glintMinDist'] < distance < sliderVals['glintMaxDist']:
                    # further filtering based on vertical position
                    vDist = abs(glint1[1] - glint2[1])
                    if vDist < 8:
                        filteredGlints.append(glints[currentGlt])
                        filteredGlints.append(glints[nextGlt])   

        # I've commented this section out as the glints are only accepted if they are
        # within the radius of the pupil. A valid glint might actually be outside the pupil though!
        
        #for pupil in pupils:
            #for glint in glints:
                #center, radius, angle = pupil
                ## Max radius of our pupil ellipse
                #max_radius = max(radius)
                
                ## Distance between center of pupil and glint
                #distance = self.__EuclideanDistance(center, glint)
                
                ## If distance is lower (meaning its within) radius of the pupil, this glint belongs to the pupil
                #if distance < max_radius: 
                    #filteredGlints.append(glint)
        
        # New pupil glint filter
        for pupil in pupils:
            pupilCentre = int(pupil[0][0]), int(pupil[0][1])
            for glint in filteredGlints:
                distance = self.__EuclideanDistance(pupilCentre, glint)
                if distance < 90:
                    matchedGlints.append(glint)
                    if pupil not in matchedPupils:
                        matchedPupils.append(pupil)        

        return matchedPupils, matchedGlints
    
    def __EuclideanDistance(self, pupil, glint):
        px, py = pupil
        gx, gy = glint
        return m.sqrt((px - gx) ** 2 + (py - gy) ** 2)

    def __GetEyeCorners(self, grayscale, leftTemplate, rightTemplate, pupilPosition=None):
        """Given two templates and the pupil center returns the eye corners position."""
        corners = []
        
        if leftTemplate != [] and rightTemplate != []:
            # Template match the templates on the image
            ccnormed_left = cv2.matchTemplate(grayscale, leftTemplate, cv2.TM_CCOEFF_NORMED)
            ccnormed_right = cv2.matchTemplate(grayscale, rightTemplate, cv2.TM_CCOEFF_NORMED)
            
            cv2.imshow("Left Template", ccnormed_left)
            cv2.imshow("Right Template", ccnormed_right)
            
            # Get upper left corner of the templates
            minVal, maxVal, minLoc, maxLoc_left_from = cv2.minMaxLoc(ccnormed_left)
            minVal, maxVal, minLoc, maxLoc_right_from = cv2.minMaxLoc(ccnormed_right)
            
            # Calculate lower right corner of the templates
            maxLoc_left_to = (maxLoc_left_from[0] + leftTemplate.shape[1], maxLoc_left_from[1] + leftTemplate.shape[0])
            maxLoc_right_to = (maxLoc_right_from[0] + rightTemplate.shape[1], maxLoc_right_from[1] + rightTemplate.shape[0])
            
            corners.append(maxLoc_left_from)
            corners.append(maxLoc_left_to)
            corners.append(maxLoc_right_from)
            corners.append(maxLoc_right_to)
        return corners

    def __GetIrisUsingThreshold(self, grayscale, threshold, minSize, maxSize):
        """Given a gray level image and the pupil candidates returns a list of iris locations."""
        iris = []

        # Create a binary image.
        val, threshold = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY_INV)
        
        cv2.imshow("Iris threshold", threshold)
        
        # Get the blob properties.
        props = SIGBTools.RegionProps()
        
        # Calculate blobs.
        _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterate through blob and calculate properties of each blob.
        for cnt in contours:
            p = props.CalcContourProperties(cnt, ["centroid", "area", "extend"])
            x,y = p["Centroid"]
            area = p["Area"]
            extend = p["Extend"]
            
            if area > minSize and area < maxSize and extend > 0.5 and extend < 1.0:
                irisEllipse = cv2.fitEllipse(cnt)
                iris.append(irisEllipse)

        return iris

    def __GetIrisUsingNormals(self, grayscale, pupil, normalLength):
        """Given a grayscale level image, the pupil candidates and the length of the normals returns a list of iris locations."""
        iris = []

        # YOUR IMPLEMENTATION HERE !!!!

        return iris

    def __GetIrisUsingSimplifyedHough(self, grayscale, pupil):
        """Given a grayscale level image and the pupil candidates returns a list of iris locations using a simplified Hough Transformation."""
        iris = []

        # YOUR IMPLEMENTATION HERE !!!!

        return iris
    
    def __getGradientImageInfo(self, I):
        # Use sobel on the x and y axis
        sobelX = cv2.Sobel(I, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(I, cv2.CV_64F, 0, 1)        
        
        # Arrays for holding information about orientation and magnitude og each gradient
        #orientation = np.zeros(I.shape)
        #magnitude = np.zeros(I.shape)
        
        # Calculate orientation and magnitude of each gradient
        #for x in range(I.shape[0]):
            #for y in range(I.shape[1]):
                #orientation[x][y] = np.arctan2(sobelY[x][y], sobelX[x][y]) * (180 / m.pi)
                #magnitude[x][y] = m.sqrt(sobelY[x][y] ** 2 + sobelX[x][y] ** 2)
        
        magnitude = cv2.magnitude(sobelX, sobelY)
        orientation = cv2.phase(sobelX, sobelY)        
        
        return sobelX, sobelY, magnitude, orientation

    def __showQuiverPlot(self, I):
        # Use sobel to get gradients
        sobelX = cv2.Sobel(I, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(I, cv2.CV_64F, 0, 1)
    
        # width and height of image. Use steps to quiver plot every third gradient
        w, h = I.shape
        wStep = 3 
        hStep = 3
    
        # use array slicing to get only every third gradient, else the computation amount is too high
        newSobelX = sobelX[0:w:wStep, 0:h:hStep]
        newSobelY = sobelY[0:w:wStep, 0:h:hStep]
    
        quiver(newSobelX, newSobelY)
        show()
     
    def __CircleTest(self, grayscale, centerPoints):
        nPts = 20
        circleRadius = 100
        P = SIGBTools.GetCircleSamples(center=centerPoints, radius=circleRadius, numPoints=nPts)
        for (x, y, dx, dy) in P:
            pointCoord = (int(x),int(y))
            cv2.circle(grayscale, pointCoord, 2, (0,0,255), 2)
            cv2.line(grayscale, pointCoord, centerPoints, (0,0,255))

            
    def __FindEllipseContour(self, img, gradientMagnitude, gradientOrientation, estimatedCenter, estimatedRadius, nPts=30):
        P = SIGBTools.GetCircleSamples(center=estimatedCenter, radius=estimatedRadius, numPoints=nPts)
        newPupil = np.zeros((nPts, 1, 2)).astype(np.float32)
        t = 0
        for (x, y, dx, dy) in P:
            # Draw normals
            pointCoord = (int(x),int(y))
            #cv2.circle(img, pointCoord, 2, (0,0,255), 2)
            #cv2.line(img, pointCoord, estimatedCenter, (0,0,255))            
            
            # < Define normalLength as some maximum distance away from initial circle >
            # < Get the endpoints of the normal -> p1 , p2 >
            normal = dx, dy
            maxPoint = self.__FindMaxGradientValueOnNormal(gradientMagnitude, gradientOrientation, pointCoord, estimatedCenter, normal)
            #cv2.circle(img, tuple(maxPoint), 2, (0,255,255), 2)
            # < store maxPoint in newPupil >
            newPupil[t] = maxPoint
            t = t + 1
            # < fitPoints to model using least squares - cv2.fitellipse(newPupil) >
        return cv2.fitEllipse(newPupil)     
    
    def __FindMaxGradientValueOnNormal(self, gradientMagnitude, gradientOrientation, p1, p2, normal):
        # Get integer coordinates on the straight line between p1 and p2.
        pts = SIGBTools.GetLineCoordinates(p1, p2)
        
        # Get magnitude and orientation values from gradients on the normal
        normalVals = gradientMagnitude[pts[:,1], pts[:,0]]
        normalOrients = gradientOrientation[pts[:,1], pts[:,0]]
        
        # Calculate angle between normal and x axis
        normalAngle = np.arctan2(normal[1], normal[0]) * (180 / m.pi)
        
        # Find the index of gradient containing best suitable magnitude and orientation
        maxIndex = 0
        maxGradient = 0.0
        
        for i in range(len(normalOrients)):
            if normalVals[i] > maxGradient:
                if abs(normalOrients[i] - normalAngle) < 20:
                    maxIndex = i
                    maxGradient = normalVals[i]
        
        # Find index of max value in normalVals
        maxIndex = np.argmax(normalVals)
        
        # return coordinate of max value in image coordinates
        return pts[maxIndex]
    

    #----------------------------------------------------------------------#
    #                        Image Display Methods                         #
    #----------------------------------------------------------------------#
    def __UpdateImage(self):
        """Calculate the image features and display the result based on the slider values."""
        # Show the original image.
        cv2.imshow("Original", self.OriginalImage)

        # Get a copy of the current original image.
        self.ResultImage = image = self.OriginalImage.copy()    

        # Extract the values of the sliders.
        sliderVals = self.__GetSliderValues()

        # Perform the eye features detector.
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Equalize the histogram
        #grayscale = cv2.equalizeHist(grayscale)
        #cv2.imshow("Histogram Equalization", grayscale)

        # Normal threshold methods for pupil, glints and iris
        pupils = self.__GetPupil(grayscale,  sliderVals["pupilThr"], sliderVals["pupilMinSize"], sliderVals["pupilMaxSize"], sliderVals["pupilMinExtend"], sliderVals["pupilMaxExtend"])
        #glints = self.__GetGlints(grayscale, sliderVals["glintThr"], sliderVals["glintMinSize"], sliderVals["glintMaxSize"])
        #pupils, glints = self.__FilterPupilGlint(pupils, glints)
        #irises = self.__GetIrisUsingThreshold(grayscale, sliderVals["irisThr"], sliderVals["irisMinSize"], sliderVals["irisMaxSize"])
        
        # Kmeans methods for finding threshold and pupils by kmeans.
        #labelIm, centroids = self.__DetectPupilKMeans(grayscale, K=4, distanceWeight=40, reSize=(70,70))
        #labelIm, centroids = self.__DetectPupilKMeans(grayscale, K=12, distanceWeight=40, reSize=(70,70))        
        #labelIm, centroids = self.__DetectPupilKMeans(grayscale, K=20, distanceWeight=40, reSize=(70,70))        
        #pupils = self.__PupilFromKMeans(labelIm, centroids, grayscale, sliderVals["pupilMinSize"], sliderVals["pupilMaxSize"], sliderVals["pupilMinExtend"], sliderVals["pupilMaxExtend"])
        #threshold = self.__ThresholdFromKMeans(centroids)
        
        # Do template matching.
        #leftTemplate  = self.LeftTemplate
        #rightTemplate = self.RightTemplate
        #corners = self.__GetEyeCorners(image, leftTemplate, rightTemplate)

        # For Iris Detection - Assignment #02 (Part 02)
        #iris = self.__CircularHough(grayscale)

        # Display results.
        x, y = 10, 20
        #self.__SetText(image, (x, y), "Frame: %d" % self.FrameNumber)

        # Print the values of the threshold.
        step = 28
        #self.__SetText(image, (x, y + step),     "pupilThr :" + str(sliderVals["pupilThr"]))
        #self.__SetText(image, (x, y + 2 * step), "pupilMinSize :" + str(sliderVals["pupilMinSize"]))
        #self.__SetText(image, (x, y + 3 * step), "pupilMaxSize :" + str(sliderVals["pupilMaxSize"]))
        #self.__SetText(image, (x, y + 4 * step), "pupilMinExtend :" + str(sliderVals["pupilMinExtend"]))
        #self.__SetText(image, (x, y + 5 * step), "pupilMaxExtend :" + str(sliderVals["pupilMaxExtend"]))        
        #self.__SetText(image, (x, y + 6 * step), "glintThr :" + str(sliderVals["glintThr"]))
        #self.__SetText(image, (x, y + 7 * step), "glintMinSize :" + str(sliderVals["glintMinSize"]))
        #self.__SetText(image, (x, y + 8 * step), "glintMaxSize :" + str(sliderVals["glintMaxSize"]))

        # Gaussian blur the image. Used for pupil detection with normals, to get better results.
        #grayscale = cv2.GaussianBlur(grayscale, (7,7), 20)

        # Get gradient magnitudes and orientations from image
        gX, gY, magnitude, orientation = self.__getGradientImageInfo(grayscale)  
        
        # Show quiver plot
        self.__showQuiverPlot(grayscale)
        
        # Uncomment these lines as your methods start to work to display the result.
        for pupil in pupils:
            # For pupil by thresholding and kmeans
            cv2.ellipse(image, pupil, (0, 255, 0), 1)
            center = int(pupil[0][0]), int(pupil[0][1])
            cv2.circle(image, center, 2, (0,0,255), 4)
            
            # For pupil by using normals
            #contour = self.__FindEllipseContour(image, magnitude, orientation, center, 70)
            #cv2.ellipse(image, contour, (0,0,255), 1)
            #cv2.circle(image, center, 2, (0,0,255), 4)
        
        #for glint in glints:
        #    center = int(glint[0]), int(glint[1])
        #    cv2.circle(image, center, 2, (255, 0, 255), 5)
            
        #for iris in irises:
        #    cv2.ellipse(image, iris, (0, 255, 0), 1)      
            #center = int(iris[0][0]), int(iris[0][1])
            #irisRadius = iris[1][0] / 2
            #contour = self.__FindEllipseContour(image, magnitude, orientation, center, irisRadius)
            #cv2.ellipse(image, contour, (0,0,255), 1)
           
        #if corners != []:
        #    left_from, left_to, right_from, right_to = corners
        #    cv2.rectangle(image, left_from , left_to, (0,255,0))
        #    cv2.rectangle(image, right_from , right_to, (0,255,0))

        # Show the final processed image.
        cv2.imshow("Results", image)

    def __SetText(self, image, (x, y), string):
        """Draw a text on input image."""
        cv2.putText(image, string, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(image, string, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

    #----------------------------------------------------------------------#
    #                        Windows Events Methods                        #
    #----------------------------------------------------------------------#
    def __SetupWindowSliders(self):
        """Define windows for displaying the results and create trackbars."""
        # Windows.
        cv2.namedWindow("Original")
        cv2.namedWindow("Threshold")
        cv2.namedWindow("Results")
        cv2.namedWindow("TrackBars", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TrackBars", 500, 600)

        # Threshold value for the pupil intensity.
        cv2.createTrackbar("pupilThr", "TrackBars",  90, 255, self.__OnSlidersChange)     
        # Threshold value for the glint intensities.
        cv2.createTrackbar("glintThr", "TrackBars", 240, 255, self.__OnSlidersChange)
        # Threshold value for the iris intensity.
        cv2.createTrackbar("irisThr", "TrackBars", 145, 255, self.__OnSlidersChange)        
        # Define the minimum and maximum areas of the pupil.
        cv2.createTrackbar("pupilMinSize", "TrackBars",  20, 200, self.__OnSlidersChange)
        cv2.createTrackbar("pupilMaxSize", "TrackBars", 200, 200, self.__OnSlidersChange)
        # Define the minimum and maximum extends of the pupil.
        cv2.createTrackbar("pupilMinExtend", "TrackBars",  1, 100, self.__OnSlidersChange)
        cv2.createTrackbar("pupilMaxExtend", "TrackBars", 100, 100, self.__OnSlidersChange)
        # Define the minimum and maximum areas of the pupil glints.
        cv2.createTrackbar("glintMinSize", "TrackBars",  1, 2000, self.__OnSlidersChange)
        cv2.createTrackbar("glintMaxSize", "TrackBars", 2000, 2000, self.__OnSlidersChange)
        # Define the minimum and maximum allowed distance between two glints
        cv2.createTrackbar('glintMinDist', 'TrackBars', 49, 100, self.__OnSlidersChange)
        cv2.createTrackbar('glintMaxDist', 'TrackBars', 51, 100, self.__OnSlidersChange)        
        # Define the minimum and maximum areas of the pupil iris.
        cv2.createTrackbar("irisMinSize", "TrackBars",  1, 2000, self.__OnSlidersChange)
        cv2.createTrackbar("irisMaxSize", "TrackBars", 2000, 2000, self.__OnSlidersChange)            
        # Value to indicate whether to run or pause the video.
        cv2.createTrackbar("Stop/Start", "TrackBars", 0, 1, self.__OnSlidersChange)

    def __GetSliderValues(self):
        """Extract the values of the sliders and return these in a dictionary."""
        sliderVals = {}

        sliderVals["pupilThr"] = cv2.getTrackbarPos("pupilThr", "TrackBars")
        sliderVals["glintThr"] = cv2.getTrackbarPos("glintThr", "TrackBars")
        sliderVals["irisThr"] = cv2.getTrackbarPos("irisThr", "TrackBars")
        sliderVals["pupilMinSize"] = 50 * cv2.getTrackbarPos("pupilMinSize",    "TrackBars")
        sliderVals["pupilMaxSize"] = 50 * cv2.getTrackbarPos("pupilMaxSize",    "TrackBars")
        sliderVals["pupilMinExtend"] = 0.01 * cv2.getTrackbarPos("pupilMinExtend", "TrackBars")
        sliderVals["pupilMaxExtend"] = 0.01 * cv2.getTrackbarPos("pupilMaxExtend", "TrackBars")
        sliderVals["glintMinSize"] = 0.1 * cv2.getTrackbarPos("glintMinSize",    "TrackBars")
        sliderVals["glintMaxSize"] = 0.1 * cv2.getTrackbarPos("glintMaxSize",    "TrackBars")
        sliderVals['glintMinDist'] = cv2.getTrackbarPos('glintMinDist', 'TrackBars')
        sliderVals['glintMaxDist'] = cv2.getTrackbarPos('glintMaxDist', 'TrackBars')        
        sliderVals["irisMinSize"] = 50 * cv2.getTrackbarPos("irisMinSize",    "TrackBars")
        sliderVals["irisMaxSize"] = 50 * cv2.getTrackbarPos("irisMaxSize",    "TrackBars")        
        sliderVals["Running"] = 1 == cv2.getTrackbarPos("Stop/Start", "TrackBars")

        return sliderVals

    def __OnSlidersChange(self, dummy=None):
        """Handle updates when slides have changed.
           This method only updates the display when the video is put on pause."""
        sliderVals = self.__GetSliderValues()
        if not sliderVals["Running"]:
            self.__UpdateImage()
