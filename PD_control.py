"""
PD controller file for 3R robot trajectory tracking
"""

import numpy as np

# PD gains (Kp, Kd), *can be tuned to test - note for us*, using PA3 as a reference
zeta = 0.85
omega_n = 4.0
Kp = omega_n**2
Kd = 2 * zeta * omega_n
# Diagonal gain matrices
Kp_matrix = np.diag([Kp, Kp, Kp])
Kd_matrix = np.diag([Kd, Kd, Kd])

# PD controller function, using PA3 as a reference
def pd_controller(q, qd, q_des, qd_des):
    """
    q being the joint angles (theta1-3), qd being current joint velocities, q_des being desired joint angles,
    qd_des being desired joint velocities
    """
    e = q_des - q # Position error
    ed = qd_des - qd # Velocity error
    # PD control law
    tau = Kp_matrix.dot(e) + Kd_matrix.dot(ed)
    return tau

# Simulating trajectory tracking, using PA3 as a reference
def simulate(trajectory, dt = 0.02):
    # Converting trajectory to array
    q_des_trajectory = np.array(trajectory)
    num_waypoints = len(q_des_trajectory)
    # Computing desired velocities
    qd_des_trajectory = np.zeros((num_waypoints, 3))
    for i in range(num_waypoints - 1):
        qd_des_trajectory[i] = (q_des_trajectory[i+1] - q_des_trajectory[i]) / dt
    qd_des_trajectory[-1] = qd_des_trajectory[-2]
    # Initialize storage arrays
    q = np.zeros((num_waypoints, 3))
    qd = np.zeros((num_waypoints, 3))
    tau_initialized = np.zeros((num_waypoints, 3))
    # Initial state
    q[0, :] = q_des_trajectory[0].copy()
    qd[0, :] = np.zeros(3)
    # Simulation loop
    for i in range(num_waypoints - 1):
        q_des = q_des_trajectory[i]
        qd_des = qd_des_trajectory[i]
        # Computing control
        tau = pd_controller(q[i], qd[i], q_des, qd_des)
        tau_initialized[i] = tau
        # Euler Integration as seen in here: https://coecsl.ece.illinois.edu/me446/ME446Lab_2.pdf
        qd[i + 1] = qd[i] + tau * dt
        q[i + 1] = q[i] + qd[i + 1] * dt
    tau_initialized[-1] = tau_initialized[-2]
    # Time array
    time = np.arange(num_waypoints) * dt
    return time, q, qd, tau_initialized, q_des_trajectory, qd_des_trajectory

# Analyze tracking errors, reference to PA3
def analyze_results(time, q_initialized, q_des_trajectory):
    # Steady-state error
    final_time_window = 0.2
    idxs = np.where(time >= time[-1] - final_time_window)[0]
    # Final desired and actual positions
    final_des = q_des_trajectory[-1]
    final_act = q_initialized[-1]
    # Calculating error in degrees
    error = np.abs(final_des - final_act)
    error_degrees = np.rad2deg(error)
    print(f"\nFinal steady-state error (deg): joint 1 = {error_degrees[0]:.6f}, joint 2 = {error_degrees[1]:.6f}, joint 3 = {error_degrees[2]:.6f}")
    print(f"Max final error (deg) = {np.max(error_degrees):.6f}")
