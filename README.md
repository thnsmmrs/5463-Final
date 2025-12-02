# Obstacle Aware Motion Planning ECE 5463 Final Project

**Project Goal**
Plan and execute collision free motion for a robot from point A → B in a 2D workspace with circular and rectangular obstacles, using sampling-based motion planning, spline trajectories, and PD control.

**TODO:**

Setup and Foundation
  - Define start and end goals (in task/joint space)
    - end goal should be randomly generated or static based on obstacle generation or user input
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
