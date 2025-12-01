import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

# ==========================================
# 1. ROBOT PARAMETERS
# ==========================================
# Link 1: Base (J1) to Shoulder (J2)
L_base_x = 2.0  
L_base_z = 2.0
# Link 2: Shoulder (J2) to Elbow (J3)
L_arm1 = 2.0
# Link 3: Elbow (J3) to Housing (J4 start)
L_arm2 = 2.0

# ==========================================
# 2. KINEMATICS FUNCTIONS
# ==========================================

def dh_transform(a, alpha, d, theta):
    """
    Standard Denavit-Hartenberg Transformation Matrix.
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
    q = [theta1, theta2, theta3, d4] (radians for thetas)
    """
    t1, t2, t3, d4 = q
    
    # --- Frame transformations based on your requirements ---
    
    # 1. Base -> Shoulder (J2)
    # Standard SCARA base: Z-axis is Up.
    T01 = dh_transform(a=L_base_x, alpha=0, d=L_base_z, theta=t1)
    
    # 2. Shoulder -> Elbow (J3)
    # We twist the Z-axis from Up (at J2) to Y-axis (at J3).
    # Rotation about X-axis by -90 degrees accomplishes this.
    T12 = dh_transform(a=L_arm1, alpha=-np.pi/2, d=0, theta=t2)
    
    # 3. Elbow -> Wrist/Housing (J4)
    # We twist the Z-axis from Y-axis (at J3) to Down (-Z) (at J4).
    # Rotation about X-axis by -90 degrees accomplishes this.
    T23 = dh_transform(a=L_arm2, alpha=-np.pi/2, d=0, theta=t3)
    
    # 4. Housing -> Tip
    # The extension happens along the local Z-axis (which is World -Z).
    T34 = dh_transform(a=0, alpha=0, d=d4, theta=0)
    
    # Compute the chain
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    
    if return_all:
        # Return all frame matrices for plotting links
        return [np.eye(4), T01, T02, T03, T04]
    else:
        # Return just the tip position [x, y, z] for IK solver
        return T04[:3, 3]

def inverse_kinematics_numerical(target_pos, seed_q):
    """
    Solves for q [t1, t2, t3, d4] to reach target_pos [x,y,z].
    Uses numerical optimization to handle the redundancy (pitch vs extension).
    """
    def cost_function(q):
        # 1. Position Error Cost: Distance between current tip and target
        current_tip = forward_kinematics(q)
        pos_error = np.linalg.norm(current_tip - target_pos)
        
        # 2. Constraint Cost: Keep arm flat (theta3 = 0) unless necessary.
        # This prevents the robot from pitching up weirdly for low targets.
        pitch_cost = 0.05 * (q[2]**2)
        
        return pos_error + pitch_cost

    # Bounds for the joints:
    # Theta1 (Base): -180 to 180
    # Theta2 (Shoulder): -180 to 180
    # Theta3 (Pitch): -90 to 90
    # d4 (Extension): 0.0 to 3.5 units
    bounds = ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi/2, np.pi/2), (0.0, 3.5))
    
    # Run optimization
    result = minimize(cost_function, seed_q, bounds=bounds, method='SLSQP', tol=1e-5)
    return result.x

# ==========================================
# 3. TRAJECTORY GENERATION
# ==========================================

# Define Targets
# Base is at Z=2.0.
targets = {
    # Home: On the floor (Z=0). Arm needs to extend down.
    'Home': np.array([5.0, 0.0, 0.0]), 
    
    # A: Higher (Z=1.5). Arm retracts.
    'A':    np.array([4.0, 2.0, 1.5]), 
    
    # B: A little higher (Z=2.5). Above shoulder height.
    'B':    np.array([3.0, -2.0, 2.5]),
    
    # C: High Pitch (Z=3.5). q
    # Must pitch Elbow (J3) up significantly to reach this height.
    'C':    np.array([1.0, 3.0, 3.5])   
}

sequence = ['Home', 'A', 'Home', 'B', 'Home', 'C', 'Home']

# Generate Waypoints
print("Generating Trajectory...")
full_trajectory_q = []
current_q = np.array([0.0, 0.0, 0.0, 2.0]) # Initial seed (Flat pose)
steps_per_move = 30

for i in range(len(sequence) - 1):
    start_target = targets[sequence[i]]
    end_target = targets[sequence[i+1]]
    
    # Create Cartesian Path (Straight line in 3D space)
    for t in np.linspace(0, 1, steps_per_move):
        # Interpolate position
        target_p = start_target + (end_target - start_target) * t
        
        # Solve IK for this point
        sol_q = inverse_kinematics_numerical(target_p, current_q)
        
        # Store solution and update seed for next step (warm start)
        full_trajectory_q.append(sol_q)
        current_q = sol_q

full_trajectory_q = np.array(full_trajectory_q)
print(f"Trajectory generated with {len(full_trajectory_q)} frames.")

# ==========================================
# 4. ANIMATION
# ==========================================
print("Starting Animation...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot elements
robot_lines = [ax.plot([], [], [], lw=4, marker='o', solid_capstyle='round')[0] for _ in range(4)]
trace_line, = ax.plot([], [], [], 'm-', lw=1, alpha=0.6, label='Tip Trace')
target_markers = []

# Colors for links: Base, Arm1, Arm2, Extension
link_colors = ['#333333', '#1f77b4', '#ff7f0e', '#9467bd']

def init():
    for line in robot_lines:
        line.set_data([], [])
        line.set_3d_properties([])
    trace_line.set_data([], [])
    trace_line.set_3d_properties([])
    return robot_lines + [trace_line]

# Trace history
history_x, history_y, history_z = [], [], []

def animate(frame_idx):
    # Get joint angles for this frame
    q = full_trajectory_q[frame_idx]
    
    # Calculate Forward Kinematics to get link positions
    frames = forward_kinematics(q, return_all=True)
    
    # Extract coordinates of all joint origins
    # coords shape: (5, 3) -> [Base, J2, J3, J4, Tip]
    coords = np.array([f[:3, 3] for f in frames])
    
    # Update Robot Links
    for j in range(4):
        robot_lines[j].set_data([coords[j,0], coords[j+1,0]], [coords[j,1], coords[j+1,1]])
        robot_lines[j].set_3d_properties([coords[j,2], coords[j+1,2]])
        robot_lines[j].set_color(link_colors[j])
        
    # Update Trace
    tip = coords[-1]
    history_x.append(tip[0])
    history_y.append(tip[1])
    history_z.append(tip[2])
    trace_line.set_data(history_x, history_y)
    trace_line.set_3d_properties(history_z)
    
    # Dynamic Title
    segment = min(frame_idx // steps_per_move, len(sequence)-2)
    target_name = sequence[segment+1]
    pitch_deg = np.degrees(q[2])
    ax.set_title(f"Target: {target_name} | Tip Z: {tip[2]:.1f} | Pitch(J3): {pitch_deg:.1f}°")
    
    return robot_lines + [trace_line]

# Scene Setup
ax.set_xlim(-2, 7)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Draw static targets
for name, pos in targets.items():
    ax.scatter(*pos, s=100, marker='X', label=f'Target {name}')
ax.legend(loc='upper left')

# Ground Plane
xx, yy = np.meshgrid(np.linspace(-2, 7, 10), np.linspace(-5, 5, 10))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

ani = FuncAnimation(fig, animate, frames=len(full_trajectory_q), init_func=init, interval=30, blit=False)

plt.show()