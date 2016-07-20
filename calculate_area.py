#-------------------------------------------------------------------------------
# Name:        calculate the area of polygon
# Purpose:
#
# Author:      Administrator
#
# Created:     20-02-2013
# Copyright:   (c) Administrator 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import math


class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y


def GetAreaOfPolyGon(points):
    '''calculate the area of polygon
       points: the vertex points of polygon,each of them is the type of Point
       return: the area of polygon'''
    area = 0
    if(len(points)<3):
        raise Exception("we need at least 3 points to calculate the area")
    p1 = points[0]
    for i in range(1,len(points)-1):
        p2 = points[1]
        p3 = points[2]
        #calculate the vector
        vecp1p2 = Point(p2.x - p1.x,p2.y - p1.y)
        vecp2p3 = Point(p3.x - p2.x,p3.y - p2.y)
        #judge if it is clockwise,return positive velue when it is clockwise,otherwise return negtive area value
        vecMult = vecp1p2.x*vecp2p3.y - vecp1p2.y*vecp2p3.x


        sign = 0
        if(vecMult>0):
            sign = 1
        elif(vecMult < 0):
            sign = -1


        triArea = GetAreaOfTriangle(p1,p2,p3)*sign
        area+=triArea


    return abs(area)


def GetAreaOfTriangle(p1,p2,p3):
    '''calculate the area of triangle'''
    area = 0
    p1p2 = GetLineLength(p1,p2)
    p2p3 = GetLineLength(p2,p3)
    p3p1 = GetLineLength(p3,p1)
    s = (p1p2 + p2p3 + p3p1)/2
    area = s*(s-p1p2)*(s-p2p3)*(s-p3p1)
    area = math.sqrt(area)
    return area


def GetLineLength(p1,p2):
    '''calculate the length of the side'''
    length = math.pow((p1.x-p2.x),2) + math.pow((p1.y-p2.y),2)
    length = math.sqrt(length)
    return length


def main():
    p1 = Point(-336,-8505)
    p2 = Point(306,496)
    p3 = Point(350,219)
    p4 = Point(355,598)
    #p5 = Point(1,3)
    points = [p1,p2,p3,p4]
    area = GetAreaOfPolyGon(points)
    print(math.ceil(area))
    #assert math.ceil(area)==1


if __name__ == '__main__':
    main()