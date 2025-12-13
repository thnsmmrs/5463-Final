# Obstacle Aware Motion Planning ECE 5463 Final Project

**Demonstration Video:**

https://github.com/thnsmmrs/CV-Based-Obstacle-Aware-Planning/blob/main/Media/Demo%20Video.mp4


![RRT Path Planning Example](https://github.com/thnsmmrs/CV-Based-Obstacle-Aware-Planning/blob/main/Media/RRT-Examples/RRT-Example4.jpg)
**RRT Path Planning Example**  

![Workspace Plot and Robot Animation](https://github.com/thnsmmrs/CV-Based-Obstacle-Aware-Planning/blob/main/Media/Plotting%20and%20Animation%20Example.jpeg)
**Workspace Plot and Robot Animation**  

## Goals and Timeline


**Project Goal**
Plan and execute collision free motion for a robot from point A to B in a 2D workspace with circular and rectangular obstacles, using sampling based motion planning (RRT), spline trajectories, and PD control.

**Timeline**

*12/2* First meeting (1 hr) Ayush, Krish, Ethan
- Chose project goals, defined todo list and timeline/scope

*12/4* (2 hr)
- Ethan developed v1 of vision element utilizing OpenCV (CV2) library to use background subtraction and pixel gradients to determine an obstacle mask containing a white background and obstacles highlighted in black for use in RRT.
  
https://github.com/Practical-CV/Simple-object-tracking-with-OpenCV/tree/master
https://www.linkedin.com/feed/update/urn:li:activity:7328949339798548481/

*12/5* (45 min)
- Ethan did research for RRT implementation
  
https://github.com/muye1202/2D_RRT_for_path_planning
https://github.com/nimRobotics/RRT/blob/master/rrt.py

*12/7* (2 hr)
- Ethan added RRT function handling into original script, developed global coordinate system based on camera resolution and used that to mark and identify start and end goals to be printed and used for planning. Integrated mouse input for setting plan goal.

https://github.com/nimRobotics/RRT/blob/master/rrt.py

*12/7* (1 hr)
- Krish did Morphology research and how to implement it i.e. cleaning up obstacle detection (https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html):
  * Researched OpenCV morphological operations for obstacle mask cleanup
  * Opening (cv2.MORPH_OPEN): Removes camera noise through erosion -> dilation
  * Closing (cv2.MORPH_CLOSE): Fills holes in obstacles through dilation -> erosion
  * Dilation: Adds safety margin around detected obstacles (2 iterations, 5x5 kernel)
  * Contour filtering: removes small artifacts below threshold (100 pixel threshold)
  * Will be implemented in the robot_kinematics.py file
  * TLDR: camera sees noise and weird artifacts -> implement filters to clean up obstacle mask -> removes small dots, fills holes, adds safety space around obstacles -> uses OpenCV morphology: Opening, Closing, Erosion, Dilation, result is a clean obstacle mask utilizing morphology

*12/8* (30 mins)
- Ethan modeled and printed camera mounting assembly for optimal planar object detection

*12/8* (30 minutes)
- Ayush researched possible implementations of RRT* path planning instead of RRT for smoother object avoidance

*12/9* (1 hour 30 mins)
- Krish worked on the robot_kinematics.py file, implementing FK and IK equations and working on the .pdf file with all of the equations and overall solving process

*12/9* Meeting 2 (3 hours 30 minutes)
- Ethan wrote RRT algorithm and fixed camera output to work correctly with krish's morphology function. Changed the cv program to handle obstacles as pixels that aren't white and use a white background as the background subtraction method often led to missed object boundaries due to lighting
- Krish worked more on FK / IK equation derivations as well as altering the inverse_kinematics function in the robot_kinematics.py file since orientation was not needed and a different approach was used instead. Also started work on a check_link_collisions function that took the robot, theta1-3, and samples per link as arguments and is intended to check all three links for any collisions
- Ayush worked on PD control integration based on PA3 submissions and worked on improvements to the FK/IK equations. Also worked on RRT alternatives such as RRT* based on other Github repositories  
  
*12/9* (1 hour)
- Ethan implemented link collision checking inside RRT function but did not result in many valid paths due to link size and small work space. Can be tested further in proper work area on thursday 12/11. If this does not work we can change RRT to generate path in joint space (theta1, theta2, theta3) rather than work space (x,y). Also added an FPS debugging capability to cv program to allow for step size (distance between RRT nodes) testing. Tuning results showed optimal step size to be 8-15 with an average FPS (new paths generated per second) of 14. For smoothest path if snapshot method is used instead rather than constantly recalculating, we can use step size of 5.

*12/10* (45 minutes)
- Ethan created most of capture frame function with returning total distance from RRT for a set num of path samples to find the best one. Still working on debugging but general framework is there and is close to implementation. We will use the captured frame to map the obstacles and best path to a 720x720 plot for easier simulation and spline interpolation, rather than the aforementioned method of printing these directly overtop each cv frame.

*12/11* Meeting 3 (3 hours 30 min)
- Krish debugged and fixed PD control file and re-uploaded the updated version to the repo, and then separated the morphology and robot_kinematics files to make total integration into the main file easier for later. Also did the first two sections of the report for the project (description of task and FK/IK method summary)
- Ethan finished debugging the captured frame code and implemented the spline interpolation function from PA2 to work with the bestPath output. Tuned the sampling number with timer in seconds, results were 20 RRT calculations = 0.8s, 50 RRT calculations = 2s. Currently we are printing the spline path over the created RRT path and also showing the start and end at that one frozen frame. For animation we need to separate these onto a separate plot with the obstacle mask and spline. Still need to worry about obstacle collisions.
- Ayush worked on implementing link collision avoidance. This was found to be not viable due to the nature of the project, and avoiding all obstacles was not realistic for the project, so the team settled on only implementing the end effector obstacle avoidance. Worked a little bit on animation

*12/11* (2 hours)
- Ethan wrote code for simulation that works as expected showing the 3R arm tracing a spline just like we saw in PA2. Some parameters like link lengths still need to be tuned. Link collision checks still not successfuly implemented, joint space still the leading solution but yet to be tried.

*12/12* (45 mins)
- Krish integrated the PD controller file into the main obstacle detection .py file

*12/12* (5 hours 15 mins)
- Ethan debugged PD controller code (unsuccesful) and created plots for steady state error vs time. Generally PD controller returned 1-2 degrees of error average across theta 1 and theta 2, showing noticeable deviations of the end effector from generated spline, so PD control plots were not used for final demonstration. Link collisions in joint space not attempted, did not know the best approach.
- Ethan filmed demonstration video, finalized sections of final report, prepped repo and submitted files.


**TODO:**

Setup and Foundation
  - Define start and end goals (in task/joint space)
    - end goal should be randomly generated or static based on obstacle generation or user input
    - if vision method, user input can be defined in a coord system on camera output
  - Decide planning method (task space vs config space)

Model
  - Define model (joint vars, workspace, links)
  - Develop equations for and implement FK/IK for 3R
  - Simulate

Environment
  - Choose obstacle generation method
    - Dynamic obstacle detection (vision/camera)
    - Real world use case (ie warehouse layouts)
    - Randomly generated geometries
  - Choose motion planning method (RRT or similar)
  - Create test environment with obstacles

Motion Planning
 - Implement chosen method
 - Visualize (simulate with obstacles+final path)

PD Control
 - Use IK to get joint space trajectory and implement PD controller
 - Simulate robot following trajectory and track error

Visualization *Final Deliverable*
 - Robot animation
 - Show obstacles, path, robot pose

Extensions
  - Optimize spline params to reduce peak acceleration and jerk (hardware preservation) ✔
  - Dynamic obstacles with known (or unknown) motion
  - Implement re-planning at fixed frequency based on collision detecting ✔
  - Basic hardware implementation (2 or 3R) or simulated 3D env
  - Extra joint var implementation at 5 or 6R, or adding other joint types
