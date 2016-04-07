#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhagen                   -->
#<!--                   SSS - Software and Systems Section                  -->
#<!--           SIGB - Introduction to Graphics and Image Analysis          -->
#<!-- File       : Assignment2.py                                           -->
#<!-- Description: Main class of Assignment #02                             -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D06 - DK-2300 - Copenhagen S    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 24/10/2015                                               -->
#<!-- Change     : 24/10/2015 - Creation of this class                      -->
#<!-- Review     : 29/03/2016 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2016032901 $"

########################################################################
import cv2
import os
import numpy as np
import warnings

from pylab import draw
from pylab import figure
from pylab import plot
from pylab import show
from pylab import subplot
from pylab import title

import SIGBTools

########################################################################
class Assignment2(object):
    """Assignment2 class is the main class of Assignment #02."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __path = "./Assignments/_02/"

    #----------------------------------------------------------------------#
    #                    Assignment2 Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Assignment2 Class Constructor."""
        warnings.simplefilter("ignore")

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def Start(self):
        """Start Assignment #02."""
        option = "-1"
        clear  = lambda: os.system("cls" if os.name == "nt" else "clear")
        while option != "0":
            clear()
            print "\n\t#################################################################"
            print "\t#                                                               #"
            print "\t#                         Assignment #02                        #"
            print "\t#                                                               #"
            print "\t#################################################################\n"
            print "\t[1] Person Map Location."
            print "\t[2] Linear Texture Mapping (Ground Floor)."
            print "\t[3] Linear Texture Mapping (Moving Objects)."
            print "\t[4] Linear Texture Mapping (Ensuring a Correctly Placed Texture Map)."
            print "\t[5] Image Augmentation on Image Reality."
            print "\t[6] Camera Calibration."
            print "\t[7] Augmentation."
            print "\t[8] Example \"ShowImageAndPlot()\" method."
            print "\t[9] Example \"SimpleTextureMap()\" method."
            print "\t[0] Back.\n"
            option = raw_input("\tSelect an option: ")

            if option == "1":
                self.__ShowFloorTrackingData()
            elif option == "2":
                self.__TextureMapGroundFloor()
            elif option == "3":
                self.__TextureMapGridSequence()
            elif option == "4":
                self.__RealisticTextureMap()
            elif option == "5":
                self.__TextureMapObjectSequence()
            elif option == "6":
                self.__CalibrateCamera()
            elif option == "7":
                self.__Augmentation()
            elif option == "8":
                self.__ShowImageAndPlot()
            elif option == "9":
                self.__SimpleTextureMap()
            elif option != "0":
                raw_input("\n\tINVALID OPTION!!!")

    #----------------------------------------------------------------------#
    #                           Solutions                                  #
    #----------------------------------------------------------------------#
    def __CalculateHomography(self):
        # Load videodata and get first image from the image
        filename = self.__path + "Videos/ITUStudent.avi"
        SIGBTools.VideoCapture(filename, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)
        img1 = SIGBTools.read()
        
        # Load second image of the overview
        img2 = cv2.imread(self.__path + "Images/ITUMap.png")
        
        homography, mousepoints = SIGBTools.GetHomographyFromMouse(img1, img2, 4)
        
        return homography
        
        
    def __DisplayTrace(self, img, square, homography):
        # Get points of the square of the feet
        p1, p2 = square
        
        # Apply homography on the points. Use only the first point to draw a circle.
        p1_map = self.__ApplyHomography(p1, homography)
        #p2_map = self.__ApplyHomography(p2, homography)
        
        # Draw a circle on the image
        cv2.circle(img, p1_map, 2, (0,0,255))
        
        # Show the image
        cv2.imshow("Trace", img)
 
        
    def __TrackPerson(self, square, homography):
        img_map = cv2.imread(self.__path + "Images/ITUMap.png")
        
        # Get points of the square of the feet
        p1, p2 = square
                
        # Apply homography on the points. Use only the first point to draw a circle.
        p1_map = self.__ApplyHomography(p1, homography)
        p2_map = self.__ApplyHomography(p2, homography)
                
        # Draw a rectangle on the image
        cv2.rectangle(img_map, p1_map, p2_map, (0,0,255))
                
        # Show the image
        cv2.imshow("Tracking", img_map)
        
        # Return image
        return img_map        
 
        
    def __ApplyHomography(self, point, homography):
        # x,y,w where w >= 1
        p = np.ones(3)
        p[0] = point[0]
        p[1] = point[1]
        
        # Calculate mapping of points
        p_prime = np.dot(homography, p)
        p_prime = p_prime * 1 / p_prime[2]
        return (int(p_prime[0]), int(p_prime[1]))

    
    def __LoadOrSaveHomography(self):
        homography = None
        try:
            homography = np.load(self.__path + "Outputs/homography1.npy")
        except IOError:
            homography = self.__CalculateHomography()
            np.save(self.__path + "Outputs/homography1.npy", homography)
        return homography
    
    def __GetHomography(self, texture, corners, idx):
        """ Inspired by getHomographyFromMouse in SIGBTools
        Calculate a homography using an image and four corner points
        """
        imagePoints = []
        m, n, d = texture.shape
        # Define corner points of texture
        imagePoints.append([(0, 0), (float(n), 0), (float(n), float(m)), (0, m)])
        # Append the corners of the texture
        
        # Define corner points of the grid
        imagePoints.append([(float(corners[idx[0], 0, 0]), float(corners[idx[0], 0, 1])),
                            (float(corners[idx[1], 0, 0]), float(corners[idx[1], 0, 1])),
                            (float(corners[idx[3], 0, 0]), float(corners[idx[3], 0, 1])),
                            (float(corners[idx[2], 0, 0]), float(corners[idx[2], 0, 1]))
                            ])
        
        # Convert to openCV format
        ip1 = np.array([[x, y] for (x, y) in imagePoints[0]])
        ip2 = np.array([[x, y] for (x, y) in imagePoints[1]])
        
        # Calculate homography
        H, mask = cv2.findHomography(ip1, ip2)
        return H, imagePoints    
        

    #----------------------------------------------------------------------#
    #                        Private Class Methods                         #
    #----------------------------------------------------------------------#
    def __ShowFloorTrackingData(self):
        # Exercise 2.01 (k)
        # Our homography
        H = self.__LoadOrSaveHomography()      
        
        # Load videodata.
        filename = self.__path + "Videos/ITUStudent.avi"
        SIGBTools.VideoCapture(filename, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)
        
        # Exercise 2.01 (i)
        # Load image to be used for the trace
        img_map = cv2.imread(self.__path + "Images/ITUMap.png")

        # Exercise 2.01 (m)
        # Map location image sequence images
        map_location_images = []

        # Load tracking data.
        dataFile = np.loadtxt(self.__path + "Inputs/trackingdata.dat")
        lenght   = dataFile.shape[0]

        # Define the boxes colors.
        boxColors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # BGR.

        # Read each frame from input video and draw the rectangules on it.
        for i in range(lenght):
            # Read the current image from a video file.
            image = SIGBTools.read()

            # Draw each color rectangule in the image.
            boxes = SIGBTools.FrameTrackingData2BoxData(dataFile[i, :])
            for j in range(3):
                box = boxes[j]
                cv2.rectangle(image, box[0], box[1], boxColors[j])
            
            # Exercise 2.01 (i)
            # Display trace. Changes the given image
            self.__DisplayTrace(img_map, boxes[1], H)
            
            # Exercise 2.01 (l)
            # Display tracking.
            img_map_track = self.__TrackPerson(boxes[1], H)
            
            # Exercise 2.01 (m)
            # Create image for MapLocation.wmv
            h1, w1 = image.shape[:2]
            h2, w2 = img_map_track.shape[:2] 
            img = np.zeros((max(h1,h2), w1+w2, 3), np.uint8)
            img[:h1, :w1] = image
            img[:h2, w1:w1+w2] = img_map_track
            map_location_images.append(img)
            
            # Show the final processed image.
            cv2.imshow("Ground Floor", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Exercise 2.01 (j)
        # Save the result of tracing
        cv2.imwrite(self.__path + "Outputs/mapImage.png", img_map)
        
        # Exercise 2.01 (m)
        # Make, write and close videowriter for MapLocation.wmv
        h, w = map_location_images[0].shape[:2]
        SIGBTools.RecordingVideos(self.__path + "Outputs/MapLocation.wmv", 30.0, (w, h))        
        for img in map_location_images:
            SIGBTools.write(img)
        SIGBTools.close()

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()

    def __TextureMapGroundFloor(self):
        """Places a texture on the ground floor for each input image."""
        # Exercise 2.02 (b)
        # Load texture
        texture = cv2.imread(self.__path + "Images/ITULogo.png")
        
        # Load videodata.
        filename = self.__path + "Videos/ITUStudent.avi"
        SIGBTools.VideoCapture(filename, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)
        
        # Load tracking data.
        dataFile = np.loadtxt(self.__path + "Inputs/trackingdata.dat")
        lenght   = dataFile.shape[0]        
        
        # Exercise 2.02 (a)
        # Load first image from image sequence for the creation of the homography
        floor_img = SIGBTools.read()
        # Our homography
        H, points = SIGBTools.GetHomographyFromMouse(texture, floor_img, -1)
        np.save(self.__path + "Outputs/homography2.npy", H)
        
        # Exercise 2.02 (b)
        # TextureMapGroundFloor.wmv image sequence images
        texture_map_images = []        
        
        # Read each frame from input video and draw the texture on it
        for i in range(lenght):
            # Read the current image from a video file.
            image = SIGBTools.read()
            
            # Exercise 2.02 (b)
            # Draw the homography transformation
            h, w    = floor_img.shape[0:2]
            overlay = cv2.warpPerspective(texture, H, (w, h))
            floor_img  = cv2.addWeighted(image, 0.9, overlay, 0.1, 0)
            texture_map_images.append(floor_img)
            cv2.imshow("Texture", floor_img)

            # Show the final processed image.
            cv2.imshow("Ground Floor", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Exercise 2.02 (b)
        # Make, write and close videowriter for TextureMapGroundFloor.wmv
        h, w = texture_map_images[0].shape[:2]
        SIGBTools.RecordingVideos(self.__path + "Outputs/TextureMapGroundFloor.wmv", 30.0, (w, h))        
        for img in texture_map_images:
            SIGBTools.write(img)
        SIGBTools.close()        

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()

    def __TextureMapGridSequence(self):
        """Skeleton for texturemapping on a video sequence."""
        # Load videodata.
        filename = self.__path + "Videos/Grid05.mp4"
        SIGBTools.VideoCapture(filename, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)

        # Load texture mapping image.
        texture = cv2.imread(self.__path + "Images/ITULogo.png")
        texture = cv2.pyrDown(texture)       

        # Define the number and ids of inner corners per a chessboard row and column.
        patternSize = (9, 6)
        idx = [0, 8, 45, 53]
        
        # Exercise 2.03 (a)
        # TextureMapGridSequence_Grid0X.wmv image sequence images
        texture_map_images = []

        # Read each frame from input video.
        while True:
            # Read the current image from a video file.
            image = SIGBTools.read()
            # Blurs an image and downsamples it.
            image = cv2.pyrDown(image)

            # Exercise 2.03 (a)
            # Finds the positions of internal corners of the chessboard.
            corners = SIGBTools.FindCorners(image)
            if corners is not None:
                H, points = self.__GetHomography(texture, corners, idx)
                
                h, w, d = image.shape
                overlay = cv2.warpPerspective(texture, H, (w, h))
                grid_img = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
                texture_map_images.append(grid_img)
                cv2.imshow("Image", grid_img)
            else:
                texture_map_images.append(image)
                cv2.imshow("Image", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Exercise 2.02 (b)
        # Make, write and close videowriter for TextureMapGroundFloor.wmv
        h, w = texture_map_images[0].shape[:2]
        SIGBTools.RecordingVideos(self.__path + "Outputs/TextureMapGridSequence_Grid05.wmv", 30.0, (w, h))        
        for img in texture_map_images:
            SIGBTools.write(img)
        SIGBTools.close()        

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()

    def __RealisticTextureMap(self):
        # Load videodata.
        filename = self.__path + "Videos/ITUStudent.avi"
        SIGBTools.VideoCapture(filename, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)

        # Load tracking data.
        dataFile = np.loadtxt(self.__path + "Inputs/trackingdata.dat")
        lenght   = dataFile.shape[0]

        # Define the boxes colors.
        boxColors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # BGR.

        # Read each frame from input video and draw the rectangules on it.
        for i in range(lenght):
            # Read the current image from a video file.
            image = SIGBTools.read()

            # Draw each color rectangule in the image.
            boxes = SIGBTools.FrameTrackingData2BoxData(dataFile[i, :])
            for j in range(3):
                box = boxes[j]
                cv2.rectangle(image, box[0], box[1], boxColors[j])

            # Show the final processed image.
            cv2.imshow("Ground Floor", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()

    def __TextureMapObjectSequence(self):
        """Poor implementation of simple TextureMap."""
        # Load videodata.
        filename = self.__path + "Videos/Scene01.mp4"
        SIGBTools.VideoCapture(filename, SIGBTools.CAMERA_VIDEOCAPTURE_640X480)
        drawContours = True

        # Load texture mapping image.
        texture = cv2.imread(self.__path + "Images/ITULogo.png")

        # Read each frame from input video.
        while True:
            # Jump for each 20 frames in the video.
            for t in range(20):
                # Read the current image from a video file.
                image = SIGBTools.read()

            # Try to detect an object in the input image.
            squares = SIGBTools.DetectPlaneObject(image)

            # Check the corner of detected object.
            for sqr in squares:
                # Do texturemap here!!!!
                # TODO
                pass

            # Draws contours outlines or filled contours.
            if drawContours and len(squares) > 0:
                cv2.drawContours(image, squares, -1, (0, 255, 0), 3)

            # Show the final processed image.
            cv2.imshow("Detection", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()

    def __CalibrateCamera(self):
        """Main function used for calibrating a common webcam."""
        # Load the camera.
        cameraID = 0
        SIGBTools.VideoCapture(cameraID, SIGBTools.CAMERA_VIDEOCAPTURE_640X480_30FPS)

        # Calibrate the connected camera.
        SIGBTools.calibrate()

        # Close all allocated resources.
        SIGBTools.release()

    def __Augmentation(self):
        """Projects an augmentation object over the chessboard pattern."""
        # Load the camera.
        cameraID = 0
        SIGBTools.VideoCapture(cameraID, SIGBTools.CAMERA_VIDEOCAPTURE_640X480_30FPS)

        # Read each frame from input camera.
        while True:
            # Read the current image from the camera.
            image = SIGBTools.read()

            # Finds the positions of internal corners of the chessboard.
            corners = SIGBTools.FindCorners(image, False)
            if corners is not None:
                pass

            # Show the final processed image.
            cv2.imshow("Augmentation", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Wait 2 seconds before finishing the method.
        cv2.waitKey(2000)

        # Close all allocated resources.
        cv2.destroyAllWindows()
        SIGBTools.release()

    def __ShowImageAndPlot(self):
        """A simple attempt to get mouse inputs and display images using pylab."""
        # Read the input image.
        image  = cv2.imread(self.__path + "Images/ITUMap.png")
        image2 = image.copy()

        # Make figure and two subplots.
        fig = figure(1)
        ax1 = subplot(1, 2, 1)
        ax2 = subplot(1, 2, 2)
        ax1.imshow(image)
        ax2.imshow(image2)
        ax1.axis("image")
        ax1.axis("off")

        # Read 5 points from the input images.
        points = fig.ginput(5)
        fig.hold("on")

        # Draw the selected points in both input images.
        for point in points:
            # Draw on matplotlib.
            subplot(1, 2, 1)
            plot(point[0], point[1], "rx")
            # Draw on opencv.
            cv2.circle(image2, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)

        # Clear axis.
        ax2.cla()
        # Show the second subplot.
        ax2.imshow(image2)
        # Update display: updates are usually deferred.
        draw()
        show()
        # Save with matplotlib and opencv.
        fig.savefig(self.__path + "Outputs/imagePyLab.png")
        cv2.imwrite(self.__path + "Outputs/imageOpenCV.png", image2)

    def __SimpleTextureMap(self):
        """Example of how linear texture mapping can be done using OpenCV."""
        # Read the input images.
        image1 = cv2.imread(self.__path + "Images/ITULogo.png")
        image2 = cv2.imread(self.__path + "Images/ITUMap.png")

        # Estimate the homography.
        H, points = SIGBTools.GetHomographyFromMouse(image1, image2, 4)

        # Draw the homography transformation.
        h, w    = image2.shape[0:2]
        overlay = cv2.warpPerspective(image1, H, (w, h))
        result  = cv2.addWeighted(image2, 0.5, overlay, 0.5, 0)

        # Show the result image.
        cv2.imshow("SimpleTextureMap", result)
        cv2.waitKey(0)

        # Close all allocated resources.
        cv2.destroyAllWindows()