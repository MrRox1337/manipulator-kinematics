# Name: Aman Mishra
# Module: PDE 4431 Robot Manipulation
# Task: Coursework 2 (Updated)
# Description: 4-DOF PRRR Robot with IK Validation and Conditional Logic.
#
# Configuration: PRRR (Prismatic Base, 3x Revolute Arm)
# Joint 1: Prismatic (Vertical Lift)
# Joint 2: Revolute  (Turret Rotation)
# Joint 3: Revolute  (Shoulder Pitch)
# Joint 4: Revolute  (Elbow/Wrist Pitch)
# 
# ===================================================================================================
#                        DENAVIT-HARTENBERG (DH) TABLE
# ===================================================================================================
#
# | Link (i) | a_i (Link Length) | alpha_i (Link Twist) | d_i (Link Offset) | theta_i (Joint Angle) |
# | :------: | :---------------: | :------------------: | :---------------: | :-------------------: |
# |     1    | 0                 |   0 deg              | d_1 * (Variable)  | 0 deg                 |
# |     2    | L_base   (1.0)    | -90 deg              | 0                 | theta_2 * (Variable)  |
# |     3    | L_arm1   (2.0)    |   0 deg              | 0                 | theta_3 * (Variable)  |
# |     4    | L_arm2   (2.0)    |   0 deg              | 0                 | theta_4 * (Variable)  |
# ====================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons
from scipy.optimize import minimize

# ==========================================
# 1. CONFIGURATION (USER EDITABLE)
# ==========================================

ROBOT_CONFIG = {
    # Link Lengths (meters)
    'L_base': 1.0,    # Horizontal offset (Link 2)
    'L_arm1': 2.0,    # Shoulder to Elbow (Link 3)
    'L_arm2': 2.0,    # Elbow to Tip (Link 4)
    
    # Joint Limits (min, max)
    'd1_limits': (0.0, 6.0),           # Prismatic Lift
    'theta_limits': (-np.pi, np.pi),   # Revolute Joints (Standard full rotation)
}

SCENE_CONFIG = {
    # Target Coordinates [x, y, z]
    'targets': {
        'Home': np.array([4.0, 0.0, 0.5]),
        'A':    np.array([1.0, 3.0, 1.5]),
        'B':    np.array([1.0, 3.0, 2.5]),
        'C':    np.array([1.0, 3.0, 5.5])  # High shelf
    },
    # Order of operations
    'sequence': ['Home', 'A', 'Home', 'B', 'Home', 'C', 'Home'],
    
    # Visual settings
    'box_colors': ['red', 'green', 'blue'],
    'link_colors': ['#888888', '#1f77b4', '#ff7f0e', '#9467bd'],
    'axis_length': 0.5  # Length of the coordinate frame arrows
}

SIM_CONFIG = {
    'steps_per_move': 30,  # Interpolation steps between targets
    'interval': 30,        # Animation interval (ms)
    'fps': 20,             # GIF Frame rate
    'tolerance': 0.1       # Reachability error tolerance (meters)
}

IK_CONFIG = {
    # Weights for the cost function
    'w_floor': 1000.0,  # Penalty for hitting floor
    'w_reg_d1': 0.001,  # Preference to minimize lift height
    'w_reg_t': 0.005,   # Preference to minimize joint angles
}

# ==========================================
# 2. KINEMATICS ENGINE
# ==========================================

def dh_transform(a, alpha, d, theta):
    """ Standard Denavit-Hartenberg (DH) Transformation Matrix. """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

def forward_kinematics(q, config=ROBOT_CONFIG, return_all=False):
    """
    Calculates the robot's pose given configuration q.
    q = [d1, theta2, theta3, theta4]
    """
    d1, t2, t3, t4 = q
    
    # Unpack lengths
    l_base = config['L_base']
    l_arm1 = config['L_arm1']
    l_arm2 = config['L_arm2']

    # --- DH Paramater Implementation ---
    # Joint 1: Prismatic Lift (d1 variable)
    T01 = dh_transform(a=0, alpha=0, d=d1, theta=0)
    
    # Joint 2: Turret Rotation (theta2 variable)
    T12 = dh_transform(a=l_base, alpha=-np.pi/2, d=0, theta=t2)
    
    # Joint 3: Shoulder Pitch (theta3 variable)
    T23 = dh_transform(a=l_arm1, alpha=0, d=0, theta=t3)
    
    # Joint 4: Elbow Pitch (theta4 variable)
    T34 = dh_transform(a=l_arm2, alpha=0, d=0, theta=t4)
    
    # Chain transformations
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    
    if return_all:
        return [np.eye(4), T01, T02, T03, T04]
    else:
        return T04[:3, 3] # Return Tip Position (x, y, z)

def inverse_kinematics(target_pos, seed_q, config=ROBOT_CONFIG, ik_params=IK_CONFIG):
    """
    Solves for q to reach target_pos using numerical optimization.
    """
    def cost_function(q):
        # 1. Forward Kinematics to find current state
        frames = forward_kinematics(q, config, return_all=True)
        z_coords = [f[2, 3] for f in frames]
        current_tip = frames[-1][:3, 3]
        
        # 2. Primary Objective: Position Error
        pos_error = np.linalg.norm(current_tip - target_pos)
        
        # 3. Constraint: Floor Avoidance (Penalty)
        floor_penalty = 0
        for z in z_coords:
            if z < 0:
                floor_penalty += ik_params['w_floor'] * (z**2)

        # 4. Regularization: Prefer natural postures
        reg_cost = ik_params['w_reg_d1'] * (q[0]**2) + ik_params['w_reg_t'] * (q[2]**2)
        
        return pos_error + floor_penalty + reg_cost

    # Extract bounds
    d1_min, d1_max = config['d1_limits']
    t_min, t_max = config['theta_limits']
    
    # Bounds: [d1, t2, t3, t4]
    bounds = ((d1_min, d1_max), (t_min, t_max), (t_min, t_max), (t_min, t_max))
    
    # Optimization
    result = minimize(cost_function, seed_q, bounds=bounds, method='SLSQP', tol=1e-4)
    return result.x

def check_reachability(target_pos, seed_q):
    """ Validates if a target is reachable within tolerance. """
    sol_q = inverse_kinematics(target_pos, seed_q)
    actual_pos = forward_kinematics(sol_q)
    error = np.linalg.norm(actual_pos - target_pos)
    return (error < SIM_CONFIG['tolerance']), sol_q

# ==========================================
# 3. TRAJECTORY GENERATION
# ==========================================

def generate_trajectory():
    """Generates the full path based on SCENE_CONFIG."""
    targets = SCENE_CONFIG['targets']
    sequence = SCENE_CONFIG['sequence']
    steps = SIM_CONFIG['steps_per_move']
    
    full_trajectory = []
    
    # Initial State
    current_q = np.array([1.0, 0.0, 0.0, 0.0])
    
    # 1. Pre-validate Reachability
    reachable_status = {}
    print("--- Validating Targets ---")
    for name, pos in targets.items():
        is_valid, _ = check_reachability(pos, current_q)
        reachable_status[name] = is_valid
        status_str = "REACHABLE" if is_valid else "UNREACHABLE (Skipping)"
        print(f"Target {name}: {status_str}")
        
    print("\n--- Generating Motion Path ---")
    
    # 2. Build Path
    for i in range(len(sequence) - 1):
        start_name = sequence[i]
        end_name = sequence[i+1]
        
        start_pos = targets[start_name]
        end_pos = targets[end_name]
        
        # Skip move if either start or end is unreachable
        valid_segment = reachable_status[start_name] and reachable_status[end_name]
        
        if not valid_segment:
             print(f"Skipping segment {start_name} -> {end_name}")
        
        for t in np.linspace(0, 1, steps):
            if valid_segment:
                # Linear Interpolation
                target_p = start_pos + (end_pos - start_pos) * t
                sol_q = inverse_kinematics(target_p, current_q)
                full_trajectory.append(sol_q)
                current_q = sol_q # Update seed
            else:
                # Hold position
                full_trajectory.append(current_q)
                
    return np.array(full_trajectory), reachable_status

# Execute generation
trajectory_q, reachability_map = generate_trajectory()
total_frames = len(trajectory_q)

# ==========================================
# 4. VISUALIZATION ENGINE
# ==========================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.2) 

# Initialize Plot Artists
robot_lines = [ax.plot([], [], [], lw=5, marker='o', solid_capstyle='round')[0] for _ in range(4)]
trace_line, = ax.plot([], [], [], 'k-', lw=1, alpha=0.3)
box_plots = [ax.plot([], [], [], linestyle='None', marker='s', markersize=10)[0] for _ in range(3)]
frame_quivers = [] # Store axis arrows

# Apply Colors
for bp, color in zip(box_plots, SCENE_CONFIG['box_colors']): bp.set_color(color)

def draw_frame_axes(ax, T, length):
    """Draws RGB arrows for X, Y, Z axes of a transformation matrix."""
    origin = T[:3, 3]
    R = T[:3, :3]
    # Draw Quivers: X(Red), Y(Green), Z(Blue)
    qx = ax.quiver(origin[0], origin[1], origin[2], R[0, 0], R[1, 0], R[2, 0], color='r', length=length, normalize=True)
    qy = ax.quiver(origin[0], origin[1], origin[2], R[0, 1], R[1, 1], R[2, 1], color='g', length=length, normalize=True)
    qz = ax.quiver(origin[0], origin[1], origin[2], R[0, 2], R[1, 2], R[2, 2], color='b', length=length, normalize=True)
    return [qx, qy, qz]

def draw_environment():
    """Draws tables and shelves."""
    def draw_plate(center, size, color, label):
        x_c, y_c, z_c = center
        w, l = size
        x = [x_c - w/2, x_c + w/2]
        y = [y_c - l/2, y_c + l/2]
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z_c - 0.05)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.5, shade=True)
        # Legs
        if z_c > 0.1:
            legs = [(x_c-w/2, y_c-l/2), (x_c+w/2, y_c-l/2),
                    (x_c-w/2, y_c+l/2), (x_c+w/2, y_c+l/2)]
            for lx, ly in legs:
                ax.plot([lx, lx], [ly, ly], [0, z_c], 'k-', lw=1, alpha=0.3)
        if label:
            ax.text(x_c, y_c, z_c, label, fontsize=9, fontweight='bold', ha='center')

    targets = SCENE_CONFIG['targets']
    draw_plate(targets['Home'], (1.5, 1.5), 'peru', "Table")
    draw_plate(targets['A'], (1.0, 1.0), 'lightgray', "Shelf A")
    draw_plate(targets['B'], (1.0, 1.0), 'lightgray', "Shelf B")
    draw_plate(targets['C'], (1.0, 1.0), 'lightgray', "Shelf C")

def generate_envelope_mesh():
    """Calculates the dome working envelope mesh."""
    l_base = ROBOT_CONFIG['L_base']
    l_total_arm = ROBOT_CONFIG['L_arm1'] + ROBOT_CONFIG['L_arm2']
    d1_max = ROBOT_CONFIG['d1_limits'][1]
    d1_min = ROBOT_CONFIG['d1_limits'][0]
    
    # Vertical Wall Part
    z_wall = np.linspace(d1_min, d1_max, 10)
    r_wall = np.full_like(z_wall, l_base + l_total_arm)
    
    # Dome Cap Part
    phi = np.linspace(0, np.pi/2, 10)
    r_dome = l_base + l_total_arm * np.cos(phi)
    z_dome = d1_max + l_total_arm * np.sin(phi)
    
    # Combine
    r_prof = np.concatenate([r_wall, r_dome])
    z_prof = np.concatenate([z_wall, z_dome])
    
    # Revolution
    theta = np.linspace(0, 2*np.pi, 40)
    Theta, Idx = np.meshgrid(theta, np.arange(len(r_prof)))
    R_grid = r_prof[Idx]
    Z_grid = z_prof[Idx]
    
    X = R_grid * np.cos(Theta)
    Y = R_grid * np.sin(Theta)
    return X, Y, Z_grid

# Setup UI and Environment
draw_environment()
env_X, env_Y, env_Z = generate_envelope_mesh()
envelope_plot = ax.plot_wireframe(env_X, env_Y, env_Z, color='blue', alpha=0.4, rstride=1, cstride=1, linewidth=0.5)
envelope_plot.set_visible(False)

# Checkbox
rax = plt.axes([0.05, 0.8, 0.25, 0.15]) 
check = CheckButtons(rax, ['Show Working Envelope'], [False])
def toggle_envelope(label):
    if label == 'Show Working Envelope':
        envelope_plot.set_visible(not envelope_plot.get_visible())
        plt.draw()
check.on_clicked(toggle_envelope)

# Animation Globals
trace_x, trace_y, trace_z = [], [], []

def init():
    for line in robot_lines: line.set_data([], []); line.set_3d_properties([])
    trace_line.set_data([], []); trace_line.set_3d_properties([])
    return robot_lines + [trace_line]

def update_boxes(frame_idx, tip_pos):
    """Handles the logic for box movement based on current segment."""
    steps = SIM_CONFIG['steps_per_move']
    segment = frame_idx // steps
    if segment >= 6: segment = 5
    
    targets = SCENE_CONFIG['targets']
    
    # Helper to determine box state
    def get_box_state(target_name, pickup_segment, drop_segment):
        # If target is unreachable, box never moves
        if not reachability_map[target_name]:
            return targets['Home'], True # Stay home
            
        if segment < pickup_segment: return targets['Home'], False # Not started
        if segment == pickup_segment: return tip_pos, True       # Being carried
        return targets[target_name], True                        # Delivered

    # Logic: 
    # Box 1: Home->A (Segment 0)
    b1_pos, _ = get_box_state('A', 0, 0) # Simple logic override for first box
    if reachability_map['A']:
        b1_pos = tip_pos if segment == 0 else targets['A']
    
    # Box 2: Home->B (Segment 2)
    b2_pos, b2_vis = get_box_state('B', 2, 2)
    
    # Box 3: Home->C (Segment 4)
    b3_pos, b3_vis = get_box_state('C', 4, 4)

    # Box 1 is always visible (starts on table)
    box_plots[0].set_data([b1_pos[0]], [b1_pos[1]])
    box_plots[0].set_3d_properties([b1_pos[2]])

    for i, (pos, vis) in enumerate([(b2_pos, b2_vis), (b3_pos, b3_vis)]):
        if vis:
            box_plots[i+1].set_data([pos[0]], [pos[1]])
            box_plots[i+1].set_3d_properties([pos[2]])
        else:
            box_plots[i+1].set_data([], []); box_plots[i+1].set_3d_properties([])

def animate(frame_idx):
    global frame_quivers
    q = trajectory_q[frame_idx]
    
    # Clear old axes
    for quiver in frame_quivers:
        quiver.remove()
    frame_quivers = []
    
    # Update Robot Links
    frames = forward_kinematics(q, return_all=True)
    coords = np.array([f[:3, 3] for f in frames])
    
    for j in range(4):
        robot_lines[j].set_data([coords[j,0], coords[j+1,0]], [coords[j,1], coords[j+1,1]])
        robot_lines[j].set_3d_properties([coords[j,2], coords[j+1,2]])
        robot_lines[j].set_color(SCENE_CONFIG['link_colors'][j])
    
    # Draw New Axes
    for T in frames:
        new_quivers = draw_frame_axes(ax, T, SCENE_CONFIG['axis_length'])
        frame_quivers.extend(new_quivers)
        
    # Update Trace
    tip_pos = coords[-1]
    if len(trace_x) == 0 or np.linalg.norm(tip_pos - np.array([trace_x[-1], trace_y[-1], trace_z[-1]])) > 0.01:
        trace_x.append(tip_pos[0]); trace_y.append(tip_pos[1]); trace_z.append(tip_pos[2])
    trace_line.set_data(trace_x, trace_y); trace_line.set_3d_properties(trace_z)
    
    # Update Boxes
    update_boxes(frame_idx, tip_pos)
    
    # Update Title
    seq = SCENE_CONFIG['sequence']
    steps = SIM_CONFIG['steps_per_move']
    seg_idx = frame_idx // steps
    target_name = seq[seg_idx+1] if seg_idx+1 < len(seq) else "End"
    
    status = "Unreachable" if (target_name in reachability_map and not reachability_map[target_name]) else "Reachable"
    ax.set_title(f"PRRR Robot | Moving to {target_name} ({status})")
    
    return robot_lines + [trace_line] + box_plots + frame_quivers

# Plot Settings
ax.set_xlim(-2, 8); ax.set_ylim(-6, 6); ax.set_zlim(0, 10)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

# Ground Plane
xx, yy = np.meshgrid(np.linspace(-2, 8, 10), np.linspace(-6, 6, 10))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

ani = FuncAnimation(fig, animate, frames=total_frames, init_func=init, interval=SIM_CONFIG['interval'], blit=False)

# --- GIF Generation (Optional) ---
# To create the GIF, uncomment the lines below and run the script.
# print("Saving GIF... (This may take a minute)")
# ani.save('prrr_manipulator.gif', writer='pillow', fps=SIM_CONFIG['fps'])
# print("GIF saved.")

plt.show()