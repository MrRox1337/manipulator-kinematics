# Name: Aman Mishra
# Module: PDE 4431 Robot Manipulation
# Task: Coursework 2 (Updated)
# Description: 4-DOF PRRR Robot with IK Validation and Conditional Logic.
# ===================================================================================================
#                        DENAVIT-HARTENBERG (DH) TABLE
# ===================================================================================================
# Configuration: PRRR (Prismatic Base, 3x Revolute Arm)
# Joint 1: Prismatic (Vertical Lift)
# Joint 2: Revolute  (Turret Rotation)
# Joint 3: Revolute  (Shoulder Pitch)
# Joint 4: Revolute  (Elbow/Wrist Pitch)
#
# | Link (i) | a_i (Link Length) | alpha_i (Link Twist) | d_i (Link Offset) | theta_i (Joint Angle) |
# | :------: | :---------------: | :------------------: | :---------------: | :-------------------: |
# |     1    | 0                 |   0 deg              | d_1 * (Variable)  | 0 deg                 |
# |     2    | L_base   (1.0)    | -90 deg              | 0                 | theta_2 * (Variable)  |
# |     3    | L_arm1   (2.0)    |   0 deg              | 0                 | theta_3 * (Variable)  |
# |     4    | L_arm2   (2.0)    |   0 deg              | 0                 | theta_4 * (Variable)  |
#
# * q = [d1, theta2, theta3, theta4]
# ====================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons
from scipy.optimize import minimize

# ==========================================
# 1. ROBOT PARAMETERS
# ==========================================
L_base = 1.0    # Horizontal offset
L_arm1 = 2.0    # Length of arm segment 1
L_arm2 = 2.0    # Length of arm segment 2

# Limits
D1_MAX = 6.0
D1_MIN = 0.0

# ==========================================
# 2. KINEMATICS FUNCTIONS
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

def forward_kinematics(q, return_all=False):
    """
    Calculates the robot's pose given configuration q.
    q = [d1 (meters), theta2 (rad), theta3 (rad), theta4 (rad)]
    """
    d1, t2, t3, t4 = q
    
    # T01: Base Lift (Prismatic)
    T01 = dh_transform(a=0, alpha=0, d=d1, theta=0)
    
    # T12: Turret (Revolute)
    T12 = dh_transform(a=L_base, alpha=-np.pi/2, d=0, theta=t2)
    
    # T23: Shoulder (Revolute)
    T23 = dh_transform(a=L_arm1, alpha=0, d=0, theta=t3)
    
    # T34: Elbow/Tip (Revolute)
    T34 = dh_transform(a=L_arm2, alpha=0, d=0, theta=t4)
    
    # Compute the overall transformation matrices
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    
    if return_all:
        return [np.eye(4), T01, T02, T03, T04]
    else:
        return T04[:3, 3]

def inverse_kinematics(target_pos, seed_q):
    """
    Solves for q [d1, t2, t3, t4] to reach target_pos [x,y,z].
    Includes a HARD constraint to prevent joints going below Z=0.
    """
    def cost_function(q):
        # 1. Forward Kinematics
        frames = forward_kinematics(q, return_all=True)
        z_coords = [f[2, 3] for f in frames]
        current_tip = frames[-1][:3, 3]
        
        # 2. Position Error
        pos_error = np.linalg.norm(current_tip - target_pos)
        
        # 3. Floor Constraint (Penalty)
        floor_penalty = 0
        for z in z_coords:
            if z < 0:
                floor_penalty += 1000 * (z**2)

        # 4. Regularization
        reg_cost = 0.001 * (q[0]**2) + 0.005 * (q[2]**2)
        
        return pos_error + floor_penalty + reg_cost

    # Joint Bounds
    bounds = ((D1_MIN, D1_MAX), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi))
    
    result = minimize(cost_function, seed_q, bounds=bounds, method='SLSQP', tol=1e-4)
    return result.x

def check_reachability(target_pos, seed_q, tolerance=0.1):
    """
    Validates if a target is reachable within tolerance.
    Returns: (is_reachable, solution_q)
    """
    sol_q = inverse_kinematics(target_pos, seed_q)
    
    # Validation: Compare FK(sol) with Target
    actual_pos = forward_kinematics(sol_q)
    error = np.linalg.norm(actual_pos - target_pos)
    
    if error < tolerance:
        return True, sol_q
    else:
        return False, sol_q

# ==========================================
# 3. TRAJECTORY GENERATION WITH VALIDATION
# ==========================================

targets = {
    'Home': np.array([4.0, 0.0, 0.5]),
    'A':    np.array([1.0, 3.0, 1.5]),
    'B':    np.array([1.0, 3.0, 2.5]),
    'C':    np.array([1.0, 3.0, 5.5])  # High shelf
}

# Pre-validate targets
reachable_status = {}
print("--- Validating Targets ---")
home_q = np.array([1.0, 0.0, 0.0, 0.0])

for name, pos in targets.items():
    is_valid, _ = check_reachability(pos, home_q)
    reachable_status[name] = is_valid
    status_str = "REACHABLE" if is_valid else "UNREACHABLE (Will skip)"
    print(f"Target {name}: {status_str}")

sequence = ['Home', 'A', 'Home', 'B', 'Home', 'C', 'Home']

full_trajectory_q = []
current_q = home_q
steps_per_move = 30

print("\nGenerating Trajectory...")
for i in range(len(sequence) - 1):
    start_name = sequence[i]
    end_name = sequence[i+1]
    
    start_target = targets[start_name]
    end_target = targets[end_name]
    
    # Logic: Only move if the destination (or origin for return) is valid.
    
    is_segment_valid = True
    if not reachable_status[start_name] or not reachable_status[end_name]:
        is_segment_valid = False
        print(f"Skipping segment {start_name} -> {end_name} due to unreachability.")
    
    for t in np.linspace(0, 1, steps_per_move):
        if is_segment_valid:
            target_p = start_target + (end_target - start_target) * t
            sol_q = inverse_kinematics(target_p, current_q)
            full_trajectory_q.append(sol_q)
            current_q = sol_q
        else:
            # Stay at Home (or last valid position)
            full_trajectory_q.append(current_q)

full_trajectory_q = np.array(full_trajectory_q)
total_frames = len(full_trajectory_q)

# ==========================================
# 4. VISUALIZATION & ENVELOPE
# ==========================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.2) 

# --- Plot Elements ---
robot_lines = [ax.plot([], [], [], lw=5, marker='o', solid_capstyle='round')[0] for _ in range(4)]
trace_line, = ax.plot([], [], [], 'k-', lw=1, alpha=0.3)
box_plots = [ax.plot([], [], [], linestyle='None', marker='s', markersize=10)[0] for _ in range(3)]
for bp, c in zip(box_plots, ['red', 'green', 'blue']): bp.set_color(c)

link_colors = ['#888888', '#1f77b4', '#ff7f0e', '#9467bd']

# --- Working Envelope Generator ---
def generate_envelope():
    L_arm_total = L_arm1 + L_arm2
    R_max_horizontal = L_base + L_arm_total
    
    z_wall = np.linspace(D1_MIN, D1_MAX, 10)
    r_wall = np.full_like(z_wall, R_max_horizontal)
    
    phi = np.linspace(0, np.pi/2, 10)
    r_dome = L_base + L_arm_total * np.cos(phi)
    z_dome = D1_MAX + L_arm_total * np.sin(phi)
    
    r_profile = np.concatenate([r_wall, r_dome])
    z_profile = np.concatenate([z_wall, z_dome])
    
    theta = np.linspace(0, 2*np.pi, 40)
    Theta, Profile_Idx = np.meshgrid(theta, np.arange(len(r_profile)))
    R_grid = r_profile[Profile_Idx]
    Z_grid = z_profile[Profile_Idx]
    
    X = R_grid * np.cos(Theta)
    Y = R_grid * np.sin(Theta)
    
    return ax.plot_wireframe(X, Y, Z_grid, color='blue', alpha=0.4, rstride=1, cstride=1, linewidth=0.5)

envelope_plot = generate_envelope()
envelope_plot.set_visible(False)

# --- Checkbox Setup ---
rax = plt.axes([0.05, 0.8, 0.25, 0.15]) 
check = CheckButtons(rax, ['Show Working Envelope'], [False])
def toggle_envelope(label):
    if label == 'Show Working Envelope':
        envelope_plot.set_visible(not envelope_plot.get_visible())
        plt.draw()
check.on_clicked(toggle_envelope)

# --- Environment Drawing ---
def draw_plate(ax, center, size, color, label=None):
    x_c, y_c, z_c = center
    width, length = size
    x = [x_c - width/2, x_c + width/2]
    y = [y_c - length/2, y_c + length/2]
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_c - 0.05)
    ax.plot_surface(X, Y, Z, color=color, alpha=0.5, shade=True)
    if z_c > 0.1:
        leg_coords = [(x_c-width/2, y_c-length/2), (x_c+width/2, y_c-length/2),
                      (x_c-width/2, y_c+length/2), (x_c+width/2, y_c+length/2)]
        for lx, ly in leg_coords:
            ax.plot([lx, lx], [ly, ly], [0, z_c], 'k-', lw=1, alpha=0.3)
    if label:
        ax.text(x_c, y_c, z_c, label, fontsize=9, fontweight='bold', ha='center')

draw_plate(ax, targets['Home'], (1.5, 1.5), 'peru', "Table")
draw_plate(ax, targets['A'], (1.0, 1.0), 'lightgray', "Shelf A")
draw_plate(ax, targets['B'], (1.0, 1.0), 'lightgray', "Shelf B")
draw_plate(ax, targets['C'], (1.0, 1.0), 'lightgray', "Shelf C")

# --- Animation ---
trace_x, trace_y, trace_z = [], [], []

def init():
    for line in robot_lines: line.set_data([], []); line.set_3d_properties([])
    trace_line.set_data([], []); trace_line.set_3d_properties([])
    return robot_lines + [trace_line]

def animate(frame_idx):
    q = full_trajectory_q[frame_idx]
    
    frames = forward_kinematics(q, return_all=True)
    coords = np.array([f[:3, 3] for f in frames])
    
    for j in range(4):
        robot_lines[j].set_data([coords[j,0], coords[j+1,0]], [coords[j,1], coords[j+1,1]])
        robot_lines[j].set_3d_properties([coords[j,2], coords[j+1,2]])
        robot_lines[j].set_color(link_colors[j])
        
    tip_pos = coords[-1]
    if len(trace_x) == 0 or np.linalg.norm(tip_pos - np.array([trace_x[-1], trace_y[-1], trace_z[-1]])) > 0.01:
        trace_x.append(tip_pos[0]); trace_y.append(tip_pos[1]); trace_z.append(tip_pos[2])
    trace_line.set_data(trace_x, trace_y); trace_line.set_3d_properties(trace_z)
    
    # --- Box Logic ---
    segment = frame_idx // steps_per_move
    if segment >= 6: segment = 5
    
    if not reachable_status['A']:
        b1_pos = targets['Home']
    else:
        b1_pos = tip_pos if segment == 0 else targets['A']
        
    if not reachable_status['B']:
        b2_pos = targets['Home']
        b2_vis = True 
    else:
        if segment < 2: b2_pos = targets['Home']; b2_vis = False
        elif segment == 2: b2_pos = tip_pos; b2_vis = True
        else: b2_pos = targets['B']; b2_vis = True
        
    if not reachable_status['C']:
        b3_pos = targets['Home']
        b3_vis = True
    else:
        if segment < 4: b3_pos = targets['Home']; b3_vis = False
        elif segment == 4: b3_pos = tip_pos; b3_vis = True
        else: b3_pos = targets['C']; b3_vis = True

    box_plots[0].set_data([b1_pos[0]], [b1_pos[1]]); box_plots[0].set_3d_properties([b1_pos[2]])
    
    if b2_vis:
        box_plots[1].set_data([b2_pos[0]], [b2_pos[1]]); box_plots[1].set_3d_properties([b2_pos[2]])
    else:
        box_plots[1].set_data([], []); box_plots[1].set_3d_properties([])

    if b3_vis:
        box_plots[2].set_data([b3_pos[0]], [b3_pos[1]]); box_plots[2].set_3d_properties([b3_pos[2]])
    else:
        box_plots[2].set_data([], []); box_plots[2].set_3d_properties([])

    current_target_name = sequence[segment+1] if segment+1 < len(sequence) else "End"
    if current_target_name in targets and not reachable_status[current_target_name]:
        status_text = f"Skipping {current_target_name} (Unreachable)"
    else:
        status_text = f"Moving to {current_target_name}"
        
    ax.set_title(f"PRRR Robot | {status_text}")
    return robot_lines + [trace_line] + box_plots

ax.set_xlim(-2, 8); ax.set_ylim(-6, 6); ax.set_zlim(0, 10)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

xx, yy = np.meshgrid(np.linspace(-2, 8, 10), np.linspace(-6, 6, 10))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

ani = FuncAnimation(fig, animate, frames=total_frames, init_func=init, interval=30, blit=False)

# --- GIF Generation ---
# To create the GIF, uncomment the lines below and run the script.
# Requires 'pillow' package (pip install pillow)
# print("Saving GIF... (This may take a minute)")
# ani.save('images/prrr_manipulator.gif', writer='pillow', fps=20)
# print("GIF saved as 'images/prrr_manipulator.gif'")

plt.show()