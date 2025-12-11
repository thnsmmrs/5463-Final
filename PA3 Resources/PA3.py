import numpy as np
import matplotlib.pyplot as plt

L1=0.5
L2=0.3

deg = np.pi/180.0
#Since np uses radians, define deg to convert degrees to radians
q_min = -90*deg
q_max = 90*deg
q_home = np.array([0.0, 0.0])
q_pick = np.array([30*deg, -20*deg])
q_place = np.array([45*deg, -30*deg])
#Givens in problem statement

#Defining time intervals (https://coecsl.ece.illinois.edu/me446/ME446Lab_2.pdf)
dt = 0.01
t_pick = 1.25        
t_hold1 = 0.25
t_place = 1.0
t_hold2 = 0.5
t_total = t_pick + t_hold1 + t_place + t_hold2
t = np.arange(0, t_total, dt)
#reverse logic of linspace (https://numpy.org/devdocs/reference/generated/numpy.arange.html)
N = len(t)

#Using PD control equation theta_double dot = K_p*(theta_desired - theta_current) 
# + K_d*(theta_dot_desired - theta_dot_current) - from above source, we can find define
# K_d and K_p based on the standard form x + 2*zeta*wn*x_dot + wn^2*x = wn^2*x_desired where
# kd = 2*zeta*wn and kp = wn^2. Note that when zeta = 1, the system is critically damped and
# assignment/Dr. Zia specified little oscillation so I will test zeta = 0.8-0.95 for desired 
# effect. Also testing wn = 6-12 rad/s for ~1 or less second response time based on the settling time
# equation analysis we did in ME3260
zeta = 0.85
omega_n = 7.0
Kp = omega_n**2
Kd = 2*zeta*omega_n

#starting position and vel
theta1 = q_home[0]
theta2 = q_home[1]
theta1_dot = 0.0
theta2_dot = 0.0

#arrays to store theta1 and theta2 values for simulation later (from our PA2)
theta1_array = np.zeros(N)
theta2_array = np.zeros(N)
theta1_dot_array = np.zeros(N)
theta2_dot_array = np.zeros(N)
ee_x_array = np.zeros(N)
ee_y_array = np.zeros(N)

theta1_desired = np.zeros(N)
theta2_desired = np.zeros(N)
#logs for desired angles (for debugging if needed)

#we want to set up a control loop that loops through each time step (defined earlier)
# and updates the theta1 and theta2 values based on the PD control law with out kp and kd
#we also need to use FK to solve the equations with the current iteration of theta so we can
#update our history tracking arrays for simulation later. We will also need to calculate velocity
#in the PD control part so we can use that equation to create a smooth trajectory (hopefully)
#I used https://github.com/rparak/2-Link_Manipulator and my own repo from PA2 
# https://github.com/thnsmmrs/PA2 as references for dynamic equations (and the FK for 2R)

for i in range(N):
    #We need to start by determining what our desired theta1 and theta2 values are based
    #on what time interval we are in (predetermined above) ie pick vs place vs pause
    tk = t[i]
    if tk < t_pick:
        q = q_pick
    elif tk < t_pick + t_hold1:
        q = q_pick
        #if we are in time intervals of pick motion or pick hold, desired position should be
        #the given q angles for picking position
    elif tk < t_pick + t_hold1 + t_place:
        q = q_place
    else:
        q = q_place
    theta1_temp = q[0]
    theta2_temp = q[1]
    #joint velocities desired should be 0 for that velocity control component 
    theta1_dot_temp = 0.0
    theta2_dot_temp = 0.0
        #Same thing but for place position and hold
    #Now that we have our desired q angles based on the time component, we need to use PD
    #controller and the euler integrations to update our theta1 and theta2 values like we saw
    #in https://coecsl.ece.illinois.edu/me446/ME446Lab_2.pdf

    e1 = theta1_temp - theta1
    e2 = theta2_temp - theta2
    e1_dot = theta1_dot_temp - theta1_dot
    e2_dot = theta2_dot_temp - theta2_dot

    #From source and L16 notes
    theta1_double_dot = Kp*e1 + Kd*e1_dot
    theta2_double_dot = Kp*e2 + Kd*e2_dot

    #Euler integration to update theta and theta dot values
    theta1_dot = theta1_dot + theta1_double_dot*dt
    theta2_dot = theta2_dot + theta2_double_dot*dt
    theta1 = theta1 + theta1_dot*dt
    theta2 = theta2 + theta2_dot*dt

    #storing values in arrays for sim (Logic from PA2)
    theta1_array[i] = theta1
    theta2_array[i] = theta2
    theta1_dot_array[i] = theta1_dot
    theta2_dot_array[i] = theta2_dot
    ee_x_array[i] = L1*np.cos(theta1) + L2*np.cos(theta1 + theta2)
    ee_y_array[i] = L1*np.sin(theta1) + L2*np.sin(theta1 + theta2)

    theta1_desired[i] = theta1_temp
    theta2_desired[i] = theta2_temp
    #logs for desired angles 

#Now that we have the theta1 and theta2 arrays with the PD control applied, we can start our
#simulation of motion similar to PA2 but without boundary conditions since it's not user input

#From PA2
fk_x1 = L1*np.cos(theta1_array)
fk_x2 = fk_x1 + L2*(np.cos(theta1_array+theta2_array))
fk_y1 = L1*np.sin(theta1_array)
fk_y2 = fk_y1 + L2*(np.sin(theta1_array+theta2_array))
#added logic for point mass position at pick location as required
q1_pointmass, q2_pointmass = q_pick
x_pointmass = L1*np.cos(q1_pointmass) + L2*np.cos(q1_pointmass + q2_pointmass)
y_pointmass = L1*np.sin(q1_pointmass) + L2*np.sin(q1_pointmass + q2_pointmass)
q1_place, q2_place = q_place
x_place = L1*np.cos(q1_place) + L2*np.cos(q1_place + q2_place)
y_place = L1*np.sin(q1_place) + L2*np.sin(q1_place + q2_place)

fig, graph = plt.subplots()
graph.set_xlim(-1*(L1+L2), L1+L2)
graph.set_ylim(-1*(L1+L2), L1+L2)
graph.set_aspect('equal')
#recommended by Adi on PA2
plt.title("2R Planar Arm Pick and Place Simulation")

link1, = graph.plot([], [])
link2, = graph.plot([], [])
trace, = graph.plot([], [])
#added point mass plot
pointmass, = graph.plot([], [], markersize=6, marker='o')

for i in range(N):
    link1.set_data([0, fk_x1[i]], [0, fk_y1[i]])
    link2.set_data([fk_x1[i], fk_x2[i]], [fk_y1[i], fk_y2[i]])
    trace.set_data(ee_x_array[:i], ee_y_array[:i])
    if t[i] < t_pick:
        mx, my = x_pointmass, y_pointmass
    else:
        mx = ee_x_array[i]
        my = ee_y_array[i]
    pointmass.set_data([mx], [my])
    plt.pause(0.001)


theta1_final_des = q_place[0]
theta2_final_des = q_place[1]

#analyzing steady state error as specified in instructions. By messing around with zeta and 
#omega_n, I was able to get e_ss_1 = 0.0376 deg and e_ss_2 = 0.025 deg
steady_state_window = 0.5 #steady state of the last hold time duration
ss_indices = t > (t_total - steady_state_window)

# convert rad -> deg and compute errors over that window
err1_deg = (theta1_array[ss_indices] - theta1_final_des) * (180.0/np.pi)
err2_deg = (theta2_array[ss_indices] - theta2_final_des) * (180.0/np.pi)

e_ss1 = np.max(np.abs(err1_deg))
e_ss2 = np.max(np.abs(err2_deg))

print("Steady-state error joint 1 (deg):", e_ss1)
print("Steady-state error joint 2 (deg):", e_ss2)

#Plotting analysis of EE trajectory showing the path taken and pick/place points throughout
#the animation
plt.figure()
plt.plot(ee_x_array, ee_y_array, label='EE path')
plt.scatter(x_pointmass , y_pointmass,  label='Pick')
plt.scatter(x_place, y_place, label='Place')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('EE Trajectory with Pick/Place')
plt.legend()

# distance to pick and place points at each time
dist_to_pick  = np.sqrt((ee_x_array - x_pointmass)**2  + (ee_y_array - y_pointmass)**2)
dist_to_place = np.sqrt((ee_x_array - x_place)**2 + (ee_y_array - y_place)**2)

plt.figure()
plt.plot(t, dist_to_pick,  label='Distance to Pick')
plt.plot(t, dist_to_place, label='Distance to Place')
plt.xlabel('time (s)')
plt.ylabel('distance (m)')
plt.title('Distance to Pick and Place Points vs Time')
plt.legend()
plt.show()
