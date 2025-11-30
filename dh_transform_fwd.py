""" DH transformation for 2-DOF RR robot (two links) """
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

# Robot parameters
L1 = 3.0  # Link 1 length (vertical)
L2 = 5.0  # Link 2 length (horizontal)


# Joint angles (change these to test different configurations)
theta1 = 180   # Base rotation (about z0)
theta2 = 120   # Second joint rotation (about z1)

# Base position (origin)
P0 = np.array([0, 0, 0, 1])  # x=0, y=0, z=0, 1

# Simultaneous sweep: theta1 and theta2 change together
# Define desired step sizes (in degrees) for each joint
theta1_step_size = 10  # degrees
theta2_step_size = 10  # degrees

# Calculate number of steps needed for each joint
theta1_num_steps = int(np.ceil(theta1 / theta1_step_size)) + 1
theta2_num_steps = int(np.ceil(theta2 / theta2_step_size)) + 1

# Use the maximum number of steps to dictate the framerate
num_frames = max(theta1_num_steps, theta2_num_steps)

theta1_sweep_deg = np.linspace(0, theta1, num_frames)
theta2_sweep_deg = np.linspace(0, theta2, num_frames)
theta1_sweep_rad = np.radians(theta1_sweep_deg)
theta2_sweep_rad = np.radians(theta2_sweep_deg)

theta_pairs = list(zip(theta1_sweep_rad, theta2_sweep_rad))

# Precompute joint and end-effector positions for all pairs
joint2_path = []
end_effector_path = []
for t1, t2 in theta_pairs:
    T01_sweep = dh_transform(a=0, alpha=np.pi/2, d=L1, theta=t1)
    T12_sweep = dh_transform(a=L2, alpha=0, d=0, theta=t2)
    T02_sweep = T01_sweep @ T12_sweep
    P1_homogeneous_sweep = T01_sweep @ P0
    P2_homogeneous_sweep = T02_sweep @ P0
    joint2_path.append(P1_homogeneous_sweep[:3])
    end_effector_path.append(P2_homogeneous_sweep[:3])
joint2_path = np.array(joint2_path)
end_effector_path = np.array(end_effector_path)


# Link 1 transformation: Base to Joint 2
T01 = dh_transform(a=0, alpha=np.pi/2, d=L1, theta=theta1)
print("Transformation T01 (Base to Joint 2):")
print(T01)
print()

# Position of Joint 2
P1_homogeneous = T01 @ P0
P1 = P1_homogeneous[:3]
print(f"Joint 2 position: ({P1[0]:.4f}, {P1[1]:.4f}, {P1[2]:.4f})")
print()

# Link 2 transformation: Joint 2 to End-effector
T12 = dh_transform(a=L2, alpha=0, d=0, theta=theta2)
print("Transformation T12 (Joint 2 to End-effector):")
print(T12)
print()

# Total transformation: Base to End-effector
T02 = T01 @ T12
print("Total Transformation T02 (Base to End-effector):")
print(T02)
print()

# Position of End-effector
P2_homogeneous = T02 @ P0
P2 = P2_homogeneous[:3]
print(f"End-effector position: ({P2[0]:.4f}, {P2[1]:.4f}, {P2[2]:.4f})")
print()

# ============================================================================
# 3D VISUALIZATION
# ============================================================================


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Animation setup
base = np.array([0, 0, 0])
link1_line, = ax.plot([], [], [], 'b-', linewidth=6, label=f'Link 1 (L={L1})')
link2_line, = ax.plot([], [], [], 'g-', linewidth=6, label=f'Link 2 (L={L2})')
ee_path_line, = ax.plot([], [], [], 'm--', linewidth=2, label='End-effector Path (θ2 sweep)')
joint1_scatter = ax.scatter([], [], [], color='red', s=200, marker='o', label='Base (Joint 1)', edgecolors='black', linewidths=2, zorder=5)
joint2_scatter = ax.scatter([], [], [], color='orange', s=200, marker='o', label='Joint 2', edgecolors='black', linewidths=2, zorder=5)
ee_scatter = ax.scatter([], [], [], color='purple', s=200, marker='s', label='End-effector', edgecolors='black', linewidths=2, zorder=5)

def init():
    link1_line.set_data([], [])
    link1_line.set_3d_properties([])
    link2_line.set_data([], [])
    link2_line.set_3d_properties([])
    ee_path_line.set_data([], [])
    ee_path_line.set_3d_properties([])
    return link1_line, link2_line, ee_path_line, joint1_scatter, joint2_scatter, ee_scatter

def animate(i):
    joint2 = joint2_path[i]
    end_effector = end_effector_path[i]
    link1_line.set_data([base[0], joint2[0]], [base[1], joint2[1]])
    link1_line.set_3d_properties([base[2], joint2[2]])
    link2_line.set_data([joint2[0], end_effector[0]], [joint2[1], end_effector[1]])
    link2_line.set_3d_properties([joint2[2], end_effector[2]])
    ee_path_line.set_data(end_effector_path[:i+1,0], end_effector_path[:i+1,1])
    ee_path_line.set_3d_properties(end_effector_path[:i+1,2])
    joint1_scatter._offsets3d = ([base[0]], [base[1]], [base[2]])
    joint2_scatter._offsets3d = ([joint2[0]], [joint2[1]], [joint2[2]])
    ee_scatter._offsets3d = ([end_effector[0]], [end_effector[1]], [end_effector[2]])
    # Update title to show current theta1 and theta2
    t1_deg = np.degrees(theta_pairs[i][0])
    t2_deg = np.degrees(theta_pairs[i][1])
    ax.set_title(f'2-DOF RR Robot\nθ1={t1_deg:.1f}°, θ2={t2_deg:.1f}°', fontsize=14, fontweight='bold')
    return link1_line, link2_line, ee_path_line, joint1_scatter, joint2_scatter, ee_scatter

# Draw coordinate frame at base
frame_scale = 1.5
ax.quiver(0, 0, 0, frame_scale, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.6)
ax.quiver(0, 0, 0, 0, frame_scale, 0, color='green', arrow_length_ratio=0.3, linewidth=2, alpha=0.6)
ax.quiver(0, 0, 0, 0, 0, frame_scale, color='blue', arrow_length_ratio=0.3, linewidth=2, alpha=0.6)

# Labels and title
ax.set_xlabel('X', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_zlabel('Z', fontsize=12, fontweight='bold')
ax.set_title(f'2-DOF RR Robot\nθ1={np.degrees(theta1):.1f}°, θ2 sweep', fontsize=14, fontweight='bold')

# Set equal aspect ratio for all points in sweep
all_points = np.vstack((base, joint2_path, end_effector_path))
max_range = np.array([np.ptp(all_points[:, 0]),
                      np.ptp(all_points[:, 1]),
                      np.ptp(all_points[:, 2])]).max() / 2.0
max_range = max(max_range, 2.0) * 1.2
mid_x = all_points[:, 0].mean()
mid_y = all_points[:, 1].mean()
mid_z = all_points[:, 2].mean()
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_box_aspect([1, 1, 1])

# Grid and legend
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

# Add text box with info
info_text = f'Robot Configuration:\nL1 = {L1}\nL2 = {L2}\nθ1 = 0° to 60°\nθ2 = 0° to 180°'
ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Animate
ani = FuncAnimation(fig, animate, frames=len(theta_pairs), init_func=init, blit=False, interval=200, repeat=True)

plt.tight_layout()
plt.show()

print("*** Animation complete! ***")
