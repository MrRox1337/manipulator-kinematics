import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def dh_transform(a, alpha, d, theta):
    """Standard (Classic) DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

# ==========================================
# 1. ROBOT PARAMETERS (All Non-Zero)
# ==========================================
L1_height = 3.0   # Vertical height of shoulder
a1_offset = 1.5   # Horizontal offset of shoulder from center
a2_length = 4.0   # Length of Upper Arm
a3_length = 2.0   # Length of Forearm Housing (Elbow to Prismatic Start)

# ==========================================
# 2. DEFINE MOVEMENT TRAJECTORY
# ==========================================
num_frames = 100

# Start: Retracted, facing forward
start_pose = np.array([0.0, 0.0, 0.0, 1.0]) 
# End: Rotated, lifted, elbow bent, fully extended
end_pose   = np.array([120.0, 45.0, -90.0, 4.0])

# Interpolation
theta1_vals = np.linspace(start_pose[0], end_pose[0], num_frames)
theta2_vals = np.linspace(start_pose[1], end_pose[1], num_frames)
theta3_vals = np.linspace(start_pose[2], end_pose[2], num_frames)
d4_vals     = np.linspace(start_pose[3], end_pose[3], num_frames)

# ==========================================
# 3. CALCULATE KINEMATICS
# ==========================================
path_points = [] # To store [p0, p1, p2, p3, p4] for every frame

P0_local = np.array([0, 0, 0, 1])

for i in range(num_frames):
    t1 = np.radians(theta1_vals[i])
    t2 = np.radians(theta2_vals[i])
    t3 = np.radians(theta3_vals[i])
    d4 = d4_vals[i]
    
    # 1. Waist (Base -> Shoulder)
    # Now includes a1_offset!
    T01 = dh_transform(a=a1_offset, alpha=np.pi/2, d=L1_height, theta=t1)
    
    # 2. Shoulder (Shoulder -> Elbow)
    T12 = dh_transform(a=a2_length, alpha=0, d=0, theta=t2)
    
    # 3. Elbow (Elbow -> Prismatic Housing)
    # Now includes a3_length!
    T23 = dh_transform(a=a3_length, alpha=np.pi/2, d=0, theta=t3)
    
    # 4. Prismatic (Housing -> Tip)
    # d4 is the variable extension
    T34 = dh_transform(a=0, alpha=0, d=d4, theta=0)
    
    # Forward Kinematics
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    
    # Extract coordinates
    p0 = P0_local[:3]
    p1 = (T01 @ P0_local)[:3]
    p2 = (T02 @ P0_local)[:3]
    p3 = (T03 @ P0_local)[:3]
    p4 = (T04 @ P0_local)[:3]
    
    path_points.append([p0, p1, p2, p3, p4])

path_points = np.array(path_points)

# ==========================================
# 4. VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Initial lines
lines = [ax.plot([], [], [], lw=4)[0] for _ in range(4)]
colors = ['k', 'b', 'g', 'm'] # Black, Blue, Green, Magenta
labels = ['Link 1 (Waist)', 'Link 2 (Upper)', 'Link 3 (Housing)', 'Link 4 (Ext)']

for line, color, label in zip(lines, colors, labels):
    line.set_color(color)
    line.set_label(label)

joints_scatter = ax.scatter([], [], [], c='orange', s=80, edgecolors='k', zorder=10)
tip_scatter = ax.scatter([], [], [], c='red', marker='^', s=100, zorder=10, label='Tip')

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    joints_scatter._offsets3d = ([], [], [])
    tip_scatter._offsets3d = ([], [], [])
    return lines + [joints_scatter, tip_scatter]

def animate(i):
    pts = path_points[i] # Shape (5, 3)
    
    # Update segments
    # Link 1: p0 -> p1
    # Link 2: p1 -> p2
    # Link 3: p2 -> p3
    # Link 4: p3 -> p4
    for j, line in enumerate(lines):
        line.set_data([pts[j,0], pts[j+1,0]], [pts[j,1], pts[j+1,1]])
        line.set_3d_properties([pts[j,2], pts[j+1,2]])
        
    # Update joints (p0, p1, p2, p3)
    joints_scatter._offsets3d = (pts[:-1, 0], pts[:-1, 1], pts[:-1, 2])
    
    # Update tip (p4)
    tip_scatter._offsets3d = ([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]])
    
    ax.set_title(f"Frame {i}: Ext={d4_vals[i]:.2f}")
    return lines + [joints_scatter, tip_scatter]

# Setup scene
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(0, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_box_aspect([1, 1, 1])

ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init, interval=50, blit=False)
plt.show()