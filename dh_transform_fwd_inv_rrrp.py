import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

# ==========================================
# 1. ROBOT PARAMETERS
# ==========================================
L_base_x = 2.0  
L_base_z = 2.0
L_arm1 = 2.0
L_arm2 = 2.0

# ==========================================
# 2. KINEMATICS FUNCTIONS
# ==========================================

def dh_transform(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

def forward_kinematics(q, return_all=False):
    t1, t2, t3, d4 = q
    
    # Base -> Shoulder
    T01 = dh_transform(a=L_base_x, alpha=0, d=L_base_z, theta=t1)
    # Shoulder -> Elbow (Twist Z -> Y)
    T12 = dh_transform(a=L_arm1, alpha=-np.pi/2, d=0, theta=t2)
    # Elbow -> Housing (Twist Y -> -Z)
    T23 = dh_transform(a=L_arm2, alpha=-np.pi/2, d=0, theta=t3)
    # Housing -> Tip (Extension along -Z)
    T34 = dh_transform(a=0, alpha=0, d=d4, theta=0)
    
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    
    if return_all:
        return [np.eye(4), T01, T02, T03, T04]
    else:
        return T04[:3, 3]

def inverse_kinematics_numerical(target_pos, seed_q):
    def cost_function(q):
        current_tip = forward_kinematics(q)
        pos_error = np.linalg.norm(current_tip - target_pos)
        pitch_cost = 0.05 * (q[2]**2) # Keep J3 flat if possible
        return pos_error + pitch_cost

    bounds = ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi/2, np.pi/2), (0.0, 3.5))
    result = minimize(cost_function, seed_q, bounds=bounds, method='SLSQP', tol=1e-5)
    return result.x

# ==========================================
# 3. TRAJECTORY GENERATION
# ==========================================

targets = {
    'Home': np.array([5.0, 0.0, 0.0]), 
    # 'A':    np.array([4.0, 2.0, 1.5]), 
    # 'B':    np.array([3.0, -2.0, 2.5]),
    'A':    np.array([1.0, 3.0, 1.5]),
    'B':    np.array([1.0, 3.0, 2.5]),
    'C':    np.array([1.0, 3.0, 3.5])   
}

# Explicit sequence: Home -> A -> Home -> B -> Home -> C -> Home
sequence = ['Home', 'A', 'Home', 'B', 'Home', 'C', 'Home']

full_trajectory_q = []
current_q = np.array([0.0, 0.0, 0.0, 2.0]) 
steps_per_move = 40 

print("Generating Trajectory...")
for i in range(len(sequence) - 1):
    start_target = targets[sequence[i]]
    end_target = targets[sequence[i+1]]
    
    # Linear interpolation
    for t in np.linspace(0, 1, steps_per_move):
        target_p = start_target + (end_target - start_target) * t
        sol_q = inverse_kinematics_numerical(target_p, current_q)
        full_trajectory_q.append(sol_q)
        current_q = sol_q

full_trajectory_q = np.array(full_trajectory_q)
total_frames = len(full_trajectory_q)
print(f"Trajectory Ready: {total_frames} frames.")

# ==========================================
# 4. ANIMATION SETUP
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Robot Visuals
robot_lines = [ax.plot([], [], [], lw=4, marker='o', solid_capstyle='round')[0] for _ in range(4)]
trace_line, = ax.plot([], [], [], 'k-', lw=1, alpha=0.3, label='Tip Path')

# Box Visuals
box1_plot, = ax.plot([], [], [], linestyle='None', marker='s', markersize=10, color='red', label='Box 1 (-> A)')
box2_plot, = ax.plot([], [], [], linestyle='None', marker='s', markersize=10, color='green', label='Box 2 (-> B)')
box3_plot, = ax.plot([], [], [], linestyle='None', marker='s', markersize=10, color='blue', label='Box 3 (-> C)')

link_colors = ['#333333', '#1f77b4', '#ff7f0e', '#9467bd']

# --- DRAW ENVIRONMENT ---
def draw_plate(ax, center, size, color, label=None):
    """Helper to draw a flat plate at a specific location"""
    x_c, y_c, z_c = center
    width, length = size
    
    # Corners
    x = [x_c - width/2, x_c + width/2]
    y = [y_c - length/2, y_c + length/2]
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_c - 0.05) # Draw slightly below target Z so box sits on top
    
    surf = ax.plot_surface(X, Y, Z, color=color, alpha=0.5, shade=True)
    
    # Add legs for shelves to ground (visual candy)
    if z_c > 0.1:
        ax.plot([x_c - width/2, x_c - width/2], [y_c - length/2, y_c - length/2], [0, z_c], 'k-', lw=1, alpha=0.3)
        ax.plot([x_c + width/2, x_c + width/2], [y_c - length/2, y_c - length/2], [0, z_c], 'k-', lw=1, alpha=0.3)
        ax.plot([x_c - width/2, x_c - width/2], [y_c + length/2, y_c + length/2], [0, z_c], 'k-', lw=1, alpha=0.3)
        ax.plot([x_c + width/2, x_c + width/2], [y_c + length/2, y_c + length/2], [0, z_c], 'k-', lw=1, alpha=0.3)
    
    if label:
        ax.text(x_c, y_c, z_c, label, fontsize=9, fontweight='bold', ha='center')

# Draw Table at Home
draw_plate(ax, targets['Home'], (1.5, 1.5), 'peru', "Table")

# Draw Shelves
draw_plate(ax, targets['A'], (1.0, 1.0), 'lightgray', "Shelf A")
draw_plate(ax, targets['B'], (1.0, 1.0), 'lightgray', "Shelf B")
draw_plate(ax, targets['C'], (1.0, 1.0), 'lightgray', "Shelf C")

# ------------------------

def init():
    for line in robot_lines:
        line.set_data([], [])
        line.set_3d_properties([])
    trace_line.set_data([], [])
    trace_line.set_3d_properties([])
    box1_plot.set_data([], [])
    box1_plot.set_3d_properties([])
    box2_plot.set_data([], [])
    box2_plot.set_3d_properties([])
    box3_plot.set_data([], [])
    box3_plot.set_3d_properties([])
    return robot_lines + [trace_line, box1_plot, box2_plot, box3_plot]

trace_x, trace_y, trace_z = [], [], []

def animate(frame_idx):
    q = full_trajectory_q[frame_idx]
    
    # --- 1. Update Robot ---
    frames = forward_kinematics(q, return_all=True)
    coords = np.array([f[:3, 3] for f in frames])
    tip_pos = coords[-1]
    
    for j in range(4):
        robot_lines[j].set_data([coords[j,0], coords[j+1,0]], [coords[j,1], coords[j+1,1]])
        robot_lines[j].set_3d_properties([coords[j,2], coords[j+1,2]])
        robot_lines[j].set_color(link_colors[j])
        
    # Trace
    trace_x.append(tip_pos[0])
    trace_y.append(tip_pos[1])
    trace_z.append(tip_pos[2])
    trace_line.set_data(trace_x, trace_y)
    trace_line.set_3d_properties(trace_z)

    # --- 2. Update Box Logic ---
    segment = frame_idx // steps_per_move
    if segment >= 6: segment = 5 

    # --- Box 1 Logic ---
    if segment == 0:
        b1_current = tip_pos # Carrying
    else:
        b1_current = targets['A'] # Dropped
    
    # --- Box 2 Logic ---
    b2_visible = True
    if segment < 2:
        b2_visible = False 
        b2_current = targets['Home']
    elif segment == 2:
        b2_current = tip_pos # Carrying
    else:
        b2_current = targets['B'] # Dropped

    # --- Box 3 Logic ---
    b3_visible = True
    if segment < 4:
        b3_visible = False 
        b3_current = targets['Home']
    elif segment == 4:
        b3_current = tip_pos # Carrying
    else:
        b3_current = targets['C'] # Dropped

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

    stage_text = sequence[segment+1] if segment+1 < len(sequence) else "Done"
    ax.set_title(f"Placing on Shelves | Next: {stage_text}")
    
    return robot_lines + [trace_line, box1_plot, box2_plot, box3_plot]

# Setup Scene
ax.set_xlim(-2, 7)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend(loc='upper left', fontsize='small')

# Ground
xx, yy = np.meshgrid(np.linspace(-2, 7, 10), np.linspace(-5, 5, 10))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

ani = FuncAnimation(fig, animate, frames=total_frames, init_func=init, interval=30, blit=False)
plt.show()