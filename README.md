# Obstacle Aware Motion Planning ECE 5463 Final Project

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
- Ethan wrote RRT algorithm and fixed camera output to work correctly with krish's morphology function.
- Krish did ___
- Ayush did ___

Left todo:
- Morphology on obstacle mask
- Contour shape detection
- RRT algorithm returning path
- Simulation and FK/IK
- Link Collision checks (RRT accounts for EE collision)

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
  - Optimize spline params to reduce peak acceleration and jerk (hardware preservation)
  - Dynamic obstacles with known (or unknown) motion
  - Implement re-planning at fixed frequency based on collision detecting
  - Basic hardware implementation (2 or 3R) or simulated 3D env
  - Extra joint var implementation at 5 or 6R, or adding other joint types
