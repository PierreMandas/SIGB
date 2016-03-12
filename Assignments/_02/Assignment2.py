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
#<!-- Review     : 24/10/2015 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2015102401 $"

########################################################################
import SIGBTools

import cv2
import os
import numpy as np
import time
import warnings

from pylab import draw
from pylab import figure
from pylab import imshow
from pylab import plot
from pylab import show
from pylab import subplot
from pylab import title

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
            print "\t[0] Back.\n"
            option = raw_input("\tSelect an option: ")

            if option != "0":
                raw_input("\n\tINVALID OPTION!!!")

    #----------------------------------------------------------------------#
    #                        Private Class Methods                         #
    #----------------------------------------------------------------------#
