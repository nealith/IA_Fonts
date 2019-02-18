import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def extractThickness(img):
    c = cv2.countNonZero(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    count = 0
    while (cv2.countNonZero(img) != 0):
        img = cv2.erode(img,element)
        count += 1

    r = 0
    if c!=0:
        r = count/c

    return r


def skeletonization(img):
    size = np.size(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    skel = np.zeros(img.shape,np.uint8)
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

    return skel


def extractLine(edges,theta,angle):
    minLineLength = 250
    #maximum allowed distance bewteen two points on a line to count them as connected
    maxLineGap = 10
    number_of_pixels_in_line = 0
    #will return several segments of the same line
    lines = cv2.HoughLinesP(edges,1,theta,80,minLineLength,maxLineGap)
    if lines is not None:

        edges=np.dstack((edges,edges,edges))
        i=0
        theta = np.zeros(lines.shape[0])




        for x1,y1,x2,y2 in lines[:,0]:
            theta[i] = math.atan2(y2-y1,x2-x1)
            cv2.line(edges,(x1,y1),(x2,y2),(0,0,255),2)
            i+=1
            number_of_pixels_in_line += math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))

    return number_of_pixels_in_line

def extract_features(path):

    img = cv2.imread(path)
    print(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 250 ,255, cv2.THRESH_TOZERO_INV)
    skel = skeletonization(binary)
    #cv2.imshow('skel',img)
    #cv2.waitKey()
    edges_binary = cv2.Canny(skel,50,200,apertureSize = 3)

    values = {
        "180":extractLine(skel,np.pi/180,"180"),
        "135":extractLine(skel,np.pi/135,"135"),
        "90":extractLine(skel,np.pi/90,"90"),
        "45":extractLine(skel,np.pi/45,"45"),
        "30":extractLine(skel,np.pi/30,"30")
    }

    number_of_pixels = cv2.countNonZero(skel)
    percent_of_line = 0
    if number_of_pixels != 0:

        percent_of_line = (values["180"] + values["135"] + values["90"] + values["45"] + values["30"])/number_of_pixels


    car = {"line":percent_of_line,"thickness":extractThickness(binary)}
    return car



#c = extract_features('test.jpg')
#print(c)
