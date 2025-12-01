import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- 1. DH Transformation Function ---
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

# --- 2. Robot Parameters ---
L1 = 3.0  # Vertical offset
L2 = 5.0  # Link length

# --- 3. Kinematics Functions ---
def forward_kinematics(theta1, theta2):
    T01 = dh_transform(a=0, alpha=np.pi/2, d=L1, theta=theta1)
    T12 = dh_transform(a=L2, alpha=0, d=0, theta=theta2)
    T02 = T01 @ T12
    P0 = np.array([0, 0, 0, 1])
    p_base = P0[:3]
    p_joint2 = (T01 @ P0)[:3]
    p_ee = (T02 @ P0)[:3]
    return p_base, p_joint2, p_ee

def inverse_kinematics(x, y, z):
    # theta1 is the azimuth angle
    theta1 = np.arctan2(y, x)
    
    # theta2 is the elevation angle
    r_xy = np.sqrt(x**2 + y**2)
    dz = z - L1
    theta2 = np.arctan2(dz, r_xy)
    
    return theta1, theta2

def get_valid_z(x, y):
    """Calculates z for a point on the reachability sphere."""
    r_sq = x**2 + y**2
    if r_sq > L2**2:
        raise ValueError(f"Point ({x}, {y}) is out of reach.")
    dz = np.sqrt(L2**2 - r_sq)
    return L1 + dz

# --- 4. Define Targets ---
home_pos = (5.0, 0.0, 3.0) 

# Calculate valid 3D coordinates for A, B, C
target_A = (3.0, 3.0, get_valid_z(3.0, 3.0))
target_B = (4.0, -2.0, get_valid_z(4.0, -2.0))
target_C = (0.0, 4.5, get_valid_z(0.0, 4.5))

targets = {
    'Home': home_pos,
    'A': target_A,
    'B': target_B,
    'C': target_C
}

sequence_names = ['Home', 'A', 'Home', 'B', 'Home', 'C', 'Home']
sequence_coords = [targets[name] for name in sequence_names]

# --- 5. Generate Trajectory ---
full_traj_th1 = []
full_traj_th2 = []
steps_per_segment = 40

for i in range(len(sequence_coords) - 1):
    start_pt = sequence_coords[i]
    end_pt = sequence_coords[i+1]
    
    # Get Joint Angles
    th1_start, th2_start = inverse_kinematics(*start_pt)
    th1_end, th2_end = inverse_kinematics(*end_pt)
    
    # Interpolate
    # We remove the last point of each segment to avoid duplication,
    # except for the very final segment of the whole path.
    if i == len(sequence_coords) - 2:
        steps = steps_per_segment + 1 
    else:
        steps = steps_per_segment
        
    t = np.linspace(0, 1, steps_per_segment+1)
    if i < len(sequence_coords) - 2:
        t = t[:-1] 
        
    seg_th1 = th1_start + (th1_end - th1_start) * t
    seg_th2 = th2_start + (th2_end - th2_start) * t
    
    full_traj_th1.extend(seg_th1)
    full_traj_th2.extend(seg_th2)

# --- 6. Animation ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

line_link1, = ax.plot([], [], [], 'b-', linewidth=5, label='Link 1')
line_link2, = ax.plot([], [], [], 'g-', linewidth=5, label='Link 2')
pt_joints = ax.scatter([], [], [], c='k', s=100)
line_trace, = ax.plot([], [], [], 'm--', linewidth=1, alpha=0.5)

# Show Targets
colors = {'Home': 'k', 'A': 'r', 'B': 'orange', 'C': 'purple'}
for name, pos in targets.items():
    ax.scatter([pos[0]], [pos[1]], [pos[2]], c=colors[name], s=100, marker='X', label=name)

trace_x, trace_y, trace_z = [], [], []

def init():
    line_link1.set_data([], [])
    line_link1.set_3d_properties([])
    line_link2.set_data([], [])
    line_link2.set_3d_properties([])
    pt_joints._offsets3d = ([], [], [])
    return line_link1, line_link2, pt_joints, line_trace

def animate(i):
    t1 = full_traj_th1[i]
    t2 = full_traj_th2[i]
    p0, p1, p2 = forward_kinematics(t1, t2)
    
    line_link1.set_data([p0[0], p1[0]], [p0[1], p1[1]])
    line_link1.set_3d_properties([p0[2], p1[2]])
    line_link2.set_data([p1[0], p2[0]], [p1[1], p2[1]])
    line_link2.set_3d_properties([p1[2], p2[2]])
    
    pt_joints._offsets3d = ([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], [p0[2], p1[2], p2[2]])
    
    trace_x.append(p2[0])
    trace_y.append(p2[1])
    trace_z.append(p2[2])
    line_trace.set_data(trace_x, trace_y)
    line_trace.set_3d_properties(trace_z)
    
    # Identify segment for title
    segment_idx = min(int(i / steps_per_segment), len(sequence_names)-2)
    curr_t = sequence_names[segment_idx]
    next_t = sequence_names[segment_idx+1]
    ax.set_title(f"Moving: {curr_t} -> {next_t}")
    
    return line_link1, line_link2, pt_joints, line_trace

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(0, 8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

ani = FuncAnimation(fig, animate, frames=len(full_traj_th1), init_func=init, interval=30, blit=False)
plt.show()