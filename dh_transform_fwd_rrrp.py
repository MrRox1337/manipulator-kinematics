import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- 1. DH Transformation Function ---
def dh_transform(a, alpha, d, theta):
    """Standard DH Matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

# --- 2. Robot Definition ---
L_base_z = 2.0  
L_base_x = 2.0  
L_arm1 = 2.0    
L_arm2 = 2.0    

def forward_kinematics_downward(q1_deg, q2_deg, q3_deg, d4_val):
    """
    FK with J3 aligned to World +Y, and J4 (Extension) aligned to World -Z.
    """
    t1, t2, t3 = np.radians([q1_deg, q2_deg, q3_deg])
    
    # Frame 1: Base Rotation (Z Up)
    T01 = dh_transform(a=L_base_x, alpha=0, d=L_base_z, theta=t1)
    
    # Frame 2: J2 (Shoulder)
    # Z_prev (Up) -> Z_new (+Y). Rotate -90 about X.
    T12 = dh_transform(a=L_arm1, alpha=-np.pi/2, d=0, theta=t2)
    
    # Frame 3: J3 (Elbow)
    # Z_prev (+Y) -> Z_new (-Z/Down). Rotate -90 about X.
    T23 = dh_transform(a=L_arm2, alpha=-np.pi/2, d=0, theta=t3)
    
    # Frame 4: Tip (Extension along Z, which is now World -Z)
    T34 = dh_transform(a=0, alpha=0, d=d4_val, theta=0)
    
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    
    return [np.eye(4), T01, T02, T03, T04]

# --- 3. Trajectory ---
# Keyframes: [J1, J2, J3, d4]
keyframes = [
    [0, 0, 0, 0],       # Home
    [45, 0, 0, 0],      # Turn Base
    [45, 30, 0, 0],     # Turn Shoulder
    [45, 30, 0, 1.5],   # Extend Down (Pick)
    [45, 30, 0, 0],     # Retract
    [90, 60, 0, 0],     # Move
    [90, 60, -30, 0],    # Pitch J3
    [90, 60, 0, 0],     # Level
    [90, 60, 0, 1.5],   # Extend Down (Place)
    [90, 60, 0, 0],     # Retract
    [0, 0, 0, 0]        # Home
]

full_traj = []
steps_per_move = 15
for i in range(len(keyframes)-1):
    start = np.array(keyframes[i])
    end = np.array(keyframes[i+1])
    for alpha in np.linspace(0, 1, steps_per_move):
        full_traj.append(start + (end - start) * alpha)
full_traj = np.array(full_traj)

# --- 4. Animation ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

lines = [ax.plot([], [], [], lw=4, marker='o')[0] for _ in range(4)]
quivers = []

def draw_frame(ax, T, length=0.8):
    origin = T[:3, 3]
    R = T[:3, :3]
    colors = ['r', 'g', 'b'] # X, Y, Z
    qs = []
    for i in range(3):
        vec = R[:, i] * length
        qs.append(ax.quiver(*origin, *vec, color=colors[i], lw=2))
    return qs

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines

def animate(i):
    global quivers
    for q_list in quivers:
        for q in q_list:
            q.remove()
    quivers = []
    
    q1, q2, q3, d4 = full_traj[i]
    frames = forward_kinematics_downward(q1, q2, q3, d4)
    coords = np.array([f[:3, 3] for f in frames])
    
    colors = ['k', 'b', 'orange', 'purple']
    labels = ['Base', 'Link1', 'Link2', 'Link3']
    
    for j in range(4):
        lines[j].set_data([coords[j,0], coords[j+1,0]], [coords[j,1], coords[j+1,1]])
        lines[j].set_3d_properties([coords[j,2], coords[j+1,2]])
        lines[j].set_color(colors[j])
        lines[j].set_label(labels[j])
        
    for f in frames[1:]:
        quivers.append(draw_frame(ax, f))
        
    ax.set_title(f"Downward Extension | Frame {i}")
    return lines

ax.set_xlim(-2, 6)
ax.set_ylim(-2, 6)
ax.set_zlim(0, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc='upper left')

ani = FuncAnimation(fig, animate, frames=len(full_traj), init_func=init, interval=50, blit=False)
plt.show()