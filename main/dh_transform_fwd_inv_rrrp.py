# Name: Aman Mishra
# Module: PDE 4431 Robot Manipulation
# Task: Coursework 2
# Lecturers: Dr. Judhi Prasetyo and Mr. Bittu Scaria
# Date: December 2025
# Description: Forward and Inverse Kinematics for a 4-DOF RRRP Robot with Pick and Place
# Visualization from a Table to 3 Shelves.
# This code uses the Denavit-Hartenberg convention for kinematic transformations.
# ===================================================================================================
#                        DENAVIT-HARTENBERG (DH) TABLE
# ===================================================================================================
# Configuration: RRRP (Downward Extension)
# Link 1: Revolute (Base)
# Link 2: Revolute (Shoulder)
# Link 3: Revolute (Elbow Pitch)
# Link 4: Prismatic (Extension)
# 
# | Link (i) | a_i (Link Length) | alpha_i (Link Twist) | d_i (Link Offset) | theta_i (Joint Angle) |
# | :------: | :---------------: | :------------------: | :---------------: | :-------------------: |
# |     1    | L_base_x (2.0)    |   0 deg              | L_base_z (2.0)    | theta_1 * (Variable)  |
# |     2    | L_arm1   (2.0)    | -90 deg              | 0                 | theta_2 * (Variable)  |
# |     3    | L_arm2   (2.0)    | -90 deg              | 0                 | theta_3 * (Variable)  |
# |     4    | 0                 |   0 deg              | d_4 * (Variable)  | 0 deg                 |
#
# * Joint variables (theta1, theta2, theta3 are solved by IK, d4 is solved by IK)
# ====================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize # Used for Numerical Inverse Kinematics

# ==========================================
# 1. ROBOT PARAMETERS
# ==========================================
L_base_x = 2.0  # Horizontal offset from J1 (Base) to J2 (Shoulder)
L_base_z = 2.0  # Vertical height offset from J1 (Base) to J2 (Shoulder)
L_arm1 = 2.0    # Length of the upper arm link (J2 to J3)
L_arm2 = 2.0    # Length of the forearm link (J3 to J4 housing)

# ==========================================
# 2. KINEMATICS FUNCTIONS
# ==========================================

def dh_transform(a, alpha, d, theta):
    """
    Standard Denavit-Hartenberg (DH) Transformation Matrix A_i-1,i.
    Parameters should be in radians where applicable (alpha, theta).
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

def forward_kinematics(q, return_all=False):
    """
    Calculates the robot's pose given joint configuration q.
    q = [theta1 (rad), theta2 (rad), theta3 (rad), d4 (units)]
    """
    t1, t2, t3, d4 = q
    
    # T01: Base (J1) -> Shoulder (J2). DH: a=L_base_x, d=L_base_z, alpha=0
    T01 = dh_transform(a=L_base_x, alpha=0, d=L_base_z, theta=t1)
    
    # T12: Shoulder (J2) -> Elbow (J3). DH: a=L_arm1, alpha=-pi/2 (Twist Z -> Y)
    T12 = dh_transform(a=L_arm1, alpha=-np.pi/2, d=0, theta=t2)
    
    # T23: Elbow (J3) -> Housing (J4). DH: a=L_arm2, alpha=-pi/2 (Twist Y -> -Z)
    T23 = dh_transform(a=L_arm2, alpha=-np.pi/2, d=0, theta=t3)
    
    # T34: Housing (J4) -> Tip. DH: d=d4 (Extension along -Z axis)
    T34 = dh_transform(a=0, alpha=0, d=d4, theta=0)
    
    # Compute the overall transformation matrices
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    
    if return_all:
        # Include base frame T00 (Identity matrix) for full link visualization
        return [np.eye(4), T01, T02, T03, T04]
    else:
        # Return just the tip position [x, y, z] for IK solver
        return T04[:3, 3]

def inverse_kinematics(target_pos, seed_q):
    """
    Solves for q [t1, t2, t3, d4] to reach target_pos [x,y,z] using numerical optimization.
    The previous solution (seed_q) is used as a 'warm start' to ensure smooth motion.
    """
    def cost_function(q):
        # 1. Position Error Cost (Primary objective to reach target)
        current_tip = forward_kinematics(q)
        pos_error = np.linalg.norm(current_tip - target_pos)
        
        # 2. Regularization Cost (Minimize Pitch)
        # Penalizes large theta3 (pitch) angles, favoring simpler solutions where possible.
        pitch_cost = 0.05 * (q[2]**2) 
        
        return pos_error + pitch_cost

    # Joint Bounds:
    # t1, t2: Full rotation (-pi to pi)
    # t3 (Pitch): Constrained to +/- 90 degrees (from DH alpha=pi/2 twist)
    # d4 (Extension): Constrained from 0.0 (retracted) to 3.5 (max extension)
    bounds = ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi/2, np.pi/2), (0.0, 3.5))
    
    # Run optimization using Sequential Least Squares Programming (SLSQP)
    result = minimize(cost_function, seed_q, bounds=bounds, method='SLSQP', tol=1e-5)
    return result.x

# ==========================================
# 3. TRAJECTORY GENERATION
# ==========================================

# Define Target Coordinates
targets = {
    # Home: On the table (Z=0). Arm is fully extended, d4 compensates for Z=2.0 base.
    'Home': np.array([5.0, 0.0, 0.0]), 
    
    # A: Shelf 1 (Z=1.5). Shelf locations are at the same (X, Y) but increasing Z.
    'A':    np.array([1.0, 3.0, 1.5]),
    
    # B: Shelf 2 (Z=2.5). Requires pitching arm up slightly to maintain IK solution.
    'B':    np.array([1.0, 3.0, 2.5]),
    
    # C: Shelf 3 (Z=3.5). Must pitch the Elbow (J3) significantly to reach this height.
    'C':    np.array([1.0, 3.0, 3.5])   
}

# Full pick-and-place sequence: Pick, Place, Return, Repeat.
sequence = ['Home', 'A', 'Home', 'B', 'Home', 'C', 'Home']

full_trajectory_q = []
current_q = np.array([0.0, 0.0, 0.0, 2.0]) # Initial configuration (t1=0, t2=0, t3=0, d4=2.0)
steps_per_move = 40 # Resolution of the path segment

print("Generating Trajectory (Solving IK for each step)...")
for i in range(len(sequence) - 1):
    start_target = targets[sequence[i]]
    end_target = targets[sequence[i+1]]
    
    # Linear interpolation in Cartesian space (Straight line movement)
    for t in np.linspace(0, 1, steps_per_move):
        # Calculate the intermediate target position
        target_p = start_target + (end_target - start_target) * t
        
        # Solve IK and store result
        sol_q = inverse_kinematics(target_p, current_q)
        full_trajectory_q.append(sol_q)
        current_q = sol_q # Update seed for the next IK solution

full_trajectory_q = np.array(full_trajectory_q)
total_frames = len(full_trajectory_q)

# ==========================================
# 4. ANIMATION SETUP AND VISUALIZATION
# ==========================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# --- Plot Elements Initialization ---
robot_lines = [ax.plot([], [], [], lw=4, marker='o', solid_capstyle='round')[0] for _ in range(4)]
trace_line, = ax.plot([], [], [], 'k-', lw=1, alpha=0.3, label='Tip Path')
frame_quivers = [] # Store quiver artists for axes

# Box Visuals (Markers representing the payloads)
box1_plot, = ax.plot([], [], [], linestyle='None', marker='s', markersize=10, color='red', label='Box 1 (-> A)')
box2_plot, = ax.plot([], [], [], linestyle='None', marker='s', markersize=10, color='green', label='Box 2 (-> B)')
box3_plot, = ax.plot([], [], [], linestyle='None', marker='s', markersize=10, color='blue', label='Box 3 (-> C)')

link_colors = ['#333333', '#1f77b4', '#ff7f0e', '#9467bd'] # Colors for [Base, J2, J3, J4]

# --- ENVIRONMENT DRAWING FUNCTION ---
def draw_plate(ax, center, size, color, label=None):
    """Draws a shelf/table surface and connecting legs."""
    x_c, y_c, z_c = center
    width, length = size
    
    # Create surface mesh
    x = [x_c - width/2, x_c + width/2]
    y = [y_c - length/2, y_c + length/2]
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_c - 0.05) # Draw slightly below target Z so box sits on top
    
    # Plot the surface
    ax.plot_surface(X, Y, Z, color=color, alpha=0.5, shade=True)
    
    # Draw legs for shelves (visual connection to ground)
    if z_c > 0.1:
        leg_coords = [
            (x_c - width/2, y_c - length/2), (x_c + width/2, y_c - length/2),
            (x_c - width/2, y_c + length/2), (x_c + width/2, y_c + length/2)
        ]
        for lx, ly in leg_coords:
            ax.plot([lx, lx], [ly, ly], [0, z_c], 'k-', lw=1, alpha=0.3)
    
    if label:
        ax.text(x_c, y_c, z_c, label, fontsize=9, fontweight='bold', ha='center')

def draw_frame_axes(ax, T, length=0.5):
    """
    Draws RGB arrows representing the X, Y, Z axes of the transformation matrix T.
    Red = X, Green = Y, Blue = Z
    """
    origin = T[:3, 3]
    R = T[:3, :3]
    
    # Draw Quivers
    # X axis (Red)
    qx = ax.quiver(origin[0], origin[1], origin[2], 
                   R[0, 0], R[1, 0], R[2, 0], 
                   color='r', length=length, normalize=True)
    # Y axis (Green)
    qy = ax.quiver(origin[0], origin[1], origin[2], 
                   R[0, 1], R[1, 1], R[2, 1], 
                   color='g', length=length, normalize=True)
    # Z axis (Blue)
    qz = ax.quiver(origin[0], origin[1], origin[2], 
                   R[0, 2], R[1, 2], R[2, 2], 
                   color='b', length=length, normalize=True)
    return [qx, qy, qz]

# Draw Table (Home position is on the floor)
draw_plate(ax, targets['Home'], (1.5, 1.5), 'peru', "Table")

# Draw Shelves
draw_plate(ax, targets['A'], (1.0, 1.0), 'lightgray', "Shelf A")
draw_plate(ax, targets['B'], (1.0, 1.0), 'lightgray', "Shelf B")
draw_plate(ax, targets['C'], (1.0, 1.0), 'lightgray', "Shelf C")

# ------------------------

def init():
    """Initializes the animation elements."""
    for line in robot_lines:
        line.set_data([], [])
        line.set_3d_properties([])
    trace_line.set_data([], [])
    trace_line.set_3d_properties([])
    # Hide boxes initially
    box1_plot.set_data([], []); box1_plot.set_3d_properties([])
    box2_plot.set_data([], []); box2_plot.set_3d_properties([])
    box3_plot.set_data([], []); box3_plot.set_3d_properties([])
    return robot_lines + [trace_line, box1_plot, box2_plot, box3_plot]

trace_x, trace_y, trace_z = [], [], []

def animate(frame_idx):
    """Updates the robot state and box positions for each frame."""
    global frame_quivers
    
    # Clear previous axes arrows
    for q in frame_quivers:
        q.remove()
    frame_quivers = []

    q = full_trajectory_q[frame_idx]
    
    # --- 1. Update Robot ---
    # Get all frames: Base, J2, J3, J4, Tip
    frames = forward_kinematics(q, return_all=True)
    coords = np.array([f[:3, 3] for f in frames]) # (5, 3) matrix of joint coordinates
    tip_pos = coords[-1]
    
    # Update link segment positions
    for j in range(4):
        robot_lines[j].set_data([coords[j,0], coords[j+1,0]], [coords[j,1], coords[j+1,1]])
        robot_lines[j].set_3d_properties([coords[j,2], coords[j+1,2]])
        robot_lines[j].set_color(link_colors[j])
        
    # Update End-effector Trace
    trace_x.append(tip_pos[0])
    trace_y.append(tip_pos[1])
    trace_z.append(tip_pos[2])
    trace_line.set_data(trace_x, trace_y)
    trace_line.set_3d_properties(trace_z)
    
    # Draw Axes for each joint
    for T in frames:
        axes = draw_frame_axes(ax, T, length=0.6)
        frame_quivers.extend(axes)

    # --- 2. Update Box Logic (Simulating Pick and Place) ---
    segment = frame_idx // steps_per_move
    if segment >= 6: segment = 5 # Clamp to the final state

    # --- Box 1 (Red) Logic: Home -> A ---
    if segment == 0:
        b1_current = tip_pos # Segment 0 (Home->A): Box moves with the gripper
    else:
        b1_current = targets['A'] # Dropped: Box stays at target A
    
    # --- Box 2 (Green) Logic: Home -> B ---
    b2_visible = True
    if segment < 2:
        b2_visible = False # Invisible until robot returns from A (segment 2 start)
        b2_current = targets['Home']
    elif segment == 2:
        b2_current = tip_pos # Segment 2 (Home->B): Box moves with the gripper
    else:
        b2_current = targets['B'] # Dropped: Box stays at target B

    # --- Box 3 (Blue) Logic: Home -> C ---
    b3_visible = True
    if segment < 4:
        b3_visible = False # Invisible until robot returns from B (segment 4 start)
        b3_current = targets['Home']
    elif segment == 4:
        b3_current = tip_pos # Segment 4 (Home->C): Box moves with the gripper
    else:
        b3_current = targets['C'] # Dropped: Box stays at target C

    # Update Box Plots
    box1_plot.set_data([b1_current[0]], [b1_current[1]])
    box1_plot.set_3d_properties([b1_current[2]])

    if b2_visible:
        box2_plot.set_data([b2_current[0]], [b2_current[1]])
        box2_plot.set_3d_properties([b2_current[2]])
    else:
        box2_plot.set_data([], [])
        box2_plot.set_3d_properties([])

    if b3_visible:
        box3_plot.set_data([b3_current[0]], [b3_current[1]])
        box3_plot.set_3d_properties([b3_current[2]])
    else:
        box3_plot.set_data([], [])
        box3_plot.set_3d_properties([])

    # Update Title
    stage_text = sequence[segment+1] if segment+1 < len(sequence) else "Task Complete"
    ax.set_title(f"Pick and Place | Next Destination: {stage_text}")
    
    return robot_lines + [trace_line, box1_plot, box2_plot, box3_plot] + frame_quivers

# Scene Setup
ax.set_xlim(-2, 7)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 6)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.legend(loc='upper left', fontsize='small')

# Ground
xx, yy = np.meshgrid(np.linspace(-2, 7, 10), np.linspace(-5, 5, 10))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

ani = FuncAnimation(fig, animate, frames=total_frames, init_func=init, interval=30, blit=False)
plt.show()