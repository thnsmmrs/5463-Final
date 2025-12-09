##From Github.com/nimRobotics/RRT

import cv2
import numpy as np
import math
import random
import argparse
import os

class Nodes: #to store created graph
    def init(self, x,y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []
    
node_list = []
debug = True #change to false for final

def rrt(frame, start, end, steps):
    h,w = frame.shape #capturing dims for coordinate system in this case 1280x720
    node_list[0] = Nodes(start[0],start[1])
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])
    
    if debug == True:
        cv2.circle(frame, (start[0],start[1]), 5,(0,0,255),thickness=3, lineType=8)
        cv2.circle(frame, (end[0],end[1]), 5,(0,0,255),thickness=3, lineType=8)
        path = "True"
    path = "Fail"

    return path
