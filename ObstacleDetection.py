import cv2
import numpy as np
import matplotlib as plt
from scipy.interpolate import CubicSpline
from RRT import rrt
from robot_kinematics import Robot3R
from Morphology import improve_obstacle_mask
import time


capture = False #creating capture variable to be used for running single frame into RRT
dispFPS = False #display FPS debug option for tuning sampling frequency using fps
#to compare runtimes
Planned = False
robot = Robot3R(link_lengths=[200, 200, 200],base_position=(10, 10))
#initializing robot params *can be changed
goal = None
path = None
bestPath = None
RRT_samples = 50 #initializing num of samples output from RRT used to pick best option
step_size = 10 #step size for distance between nodes in RRT
#tuning notes: FPS average remained same between 25-200 and only saw slight
#drop off at 10. At sample size 5 avg fps = 13 and 10 fps = 15. 25 fps ~= 16
#global vars needed for scope

def SplineInterpolate(path,goal):
    #Inspired by PA2. Spline generation was changed a little bit to sort the array first instead here and not use arc parameters since the data
    #set is very large and caused weird overshoot behavior
    path = np.asarray(path, dtype=float)
    x = path[:, 0]
    y = path[:, 1]
    gx,gy = goal
    n = len(x)
    t = np.arange(n)
    csx = CubicSpline(t, x, bc_type="natural")
    csy = CubicSpline(t, y, bc_type="natural")

    t_new = np.linspace(0, n - 1, 500)
    x_spline = csx(t_new)
    y_spline = csy(t_new)
    x_spline[len(x_spline)-1] = gx
    y_spline[len(y_spline)-1] = gy

    return x_spline, y_spline

def mouse_callback(event, x, y, flag, params): 
    global goal
    if event == cv2.EVENT_LBUTTONDOWN:
        goal = (x, y)
#function and RRT courtesy of https://github.com/nimRobotics/RRT/blob/master/rrt.py
cv2.namedWindow("Overlay")
cv2.setMouseCallback("Overlay", mouse_callback)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#Locks resolution at 720p (from logi c270 - stays same for all cameras)

start = (50,50)
#define home position (can be in terms of h,w for ratios)

Threshold = 30  #Threshold identifies what is an object and what is not, 20-30 suggested range from doc but will be testable
overlayStrength = 0.3 #Strength of overlay color on detected obstacles, 0-1 range

if dispFPS:
    prev_time = time.time()
    fps = 0.0
    fpsarr = []
while True:
    ret, frame = cap.read()
    
    #Reads next frames continously
    if not ret:
        break
    blur = cv2.GaussianBlur(frame, (5,5), 0)
    color_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(color_hsv, (0,0,50), (180,40,255))
    fg_mask = cv2.bitwise_not(white)

    #implementing improve obstacle mask function to clean up mask using morphology
    obstacle_mask = fg_mask.copy()
    obstacle_mask = improve_obstacle_mask(obstacle_mask)
    obstacle_mask = cv2.bitwise_not(obstacle_mask)
    overlay = frame.copy()
    overlay[obstacle_mask == 255] = (255, 255, 255) #sets obstacles that are black to white for overlay
    vis = cv2.addWeighted(frame, overlayStrength, overlay, 1 - overlayStrength, 0)
    cv2.circle(vis, start, 3, (0,0,255), -1) #params as defined in circle function
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        capture = True
    if key == 27: #ESC to end program
        if dispFPS:
            avgfps = sum(fpsarr) / len(fpsarr)
            print(f"Average FPS: ", avgfps)
        cap.release()
        cv2.destroyAllWindows()
        break

    if goal != None:
        cv2.circle(vis, goal, 3, (0,0,255), -1)
    #drawing start point and goal point
    #Overlay outputs the detected obstacles on the original input frame
        if capture == True and Planned == False:
            compDist = 10000 #random arbitrary number to compare to, to find shortest path
            capturedFrame = frame.copy()
            capturedOM = obstacle_mask.copy()
            #before_loop = time.time() #debugging allowing for time tracking of RRT samples 
            for i in range(RRT_samples): 
                result = rrt(capturedOM, start, goal, step_size, robot)
                if result is None:
                    continue
                [path,dist] = result
                if dist < compDist:
                    compDist = dist
                    bestPath = path
            if bestPath is None:
                print("No path found")
                capture = False
                Planned = False
            else: 
                capture = False
                Planned = True
    if Planned is True and bestPath is not None:
        #after = time.time() #debugging for tuning sampling with time results. 20 samples = 0.8s and 50 samples = 2s
        #print(after-before_loop)
        print(bestPath)
        for (x, y) in bestPath:
            cv2.circle(vis, (x, y), 3, (255, 0, 0), -1)
        cv2.circle(vis, start, 3, (0, 0, 255), -1)
        cv2.circle(vis, goal, 3, (0, 0, 255), -1)
        x_spline, y_spline = SplineInterpolate(bestPath,goal)


        #polyline solution from AI - change or remove
        pts = np.stack([x_spline, y_spline], axis=1).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=False, color=(0, 255, 0), thickness=2)



        cv2.imshow("Overlay",vis)
        cv2.waitKey(0)
        break

    if dispFPS:
        curr_time = time.time()
        diff = curr_time - prev_time
        prev_time = curr_time

        if diff > 0:
            fps = 1.0/diff
        cv2.putText(vis,f"FPS: {fps:.1f}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 255, 0),2,cv2.LINE_AA)
        fpsarr.append(fps)
        if len(fpsarr) > 1000:
            fpsarr.pop(0) #pops first fps value if array gets too big (long runtime)

    cv2.imshow("Overlay", vis)
    #cv2.imshow("Obstacle Mask", obstacle_mask)

    #Ready for simulation