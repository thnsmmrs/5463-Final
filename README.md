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

Left todo:
- Morphology on obstacle mask
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
