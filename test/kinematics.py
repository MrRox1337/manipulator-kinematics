"""
Computes the forward/inverse kinematics for a 3-DOF RRR robot with twists.
"""

import numpy as np
import matplotlib.pyplot as plt


def forward_kinematics_3dof_rrr(theta1, theta2, theta3, l1, l2, l3, alpha1, alpha2, alpha3):
    """
    Computes the forward kinematics for a 3-DOF RRR robot with twists.
    Returns the end-effector position.
    """
    # Link 1 end (fixed base at origin)
    x1, y1, z1 = 0, 0, l1

    # Link 2 end (Joint 1 to Joint 2)
    x2 = x1 + l2 * np.cos(theta1) * np.cos(theta2 + alpha2)
    y2 = y1 + l2 * np.sin(theta1) * np.cos(theta2 + alpha2)
    z2 = z1 + l2 * np.sin(theta2 + alpha2)

    # Link 3 end (Joint 2 to End-effector)
    x3 = x2 + l3 * np.cos(theta1) * np.cos(theta2 + theta3 + alpha3)
    y3 = y2 + l3 * np.sin(theta1) * np.cos(theta2 + theta3 + alpha3)
    z3 = z2 + l3 * np.sin(theta2 + theta3 + alpha3)

    return x3, y3, z3


def inverse_kinematics_3dof_rrr(x_target, y_target, z_target, l1, l2, l3, alpha1, alpha2, alpha3):
    """
    Solves inverse kinematics for a 3-DOF RRR robot in 3D space with configurable joint twists.
    Ensures the end-effector reaches the target position exactly.
    """
    # Solve for theta1 (base rotation)
    theta1 = np.arctan2(y_target, x_target)

    # Transform the target to the arm's operational plane
    r_xy = np.sqrt(x_target**2 + y_target**2)  # Distance in xy-plane
    z_plane = z_target - l1  # Adjust for Link 1's height
    r = np.sqrt(r_xy**2 + z_plane**2)  # Effective distance from the base to the target

    # Check reachability
    if r > l2 + l3 or r < abs(l2 - l3):
        raise ValueError("Target is outside the robot's reachable workspace.")

    # Solve for theta3 (elbow angle) using the law of cosines
    cos_theta3 = (r**2 - l2**2 - l3**2) / (2 * l2 * l3)
    theta3 = np.arccos(cos_theta3)  # Elbow-up configuration

    # Solve for theta2 (shoulder angle) using trigonometry
    k1 = l2 + l3 * np.cos(theta3)
    k2 = l3 * np.sin(theta3)
    phi = np.arctan2(z_plane, r_xy)
    theta2 = phi - np.arctan2(k2, k1)

    # Adjust for joint twists
    theta2 -= alpha2
    theta3 -= alpha3

    return theta1, theta2, theta3


def validate_solution(x_target, y_target, z_target, theta1, theta2, theta3, l1, l2, l3, alpha1, alpha2, alpha3):
    """
    Validates that the forward kinematics of the solution matches the target position.
    """
    x_end, y_end, z_end = forward_kinematics_3dof_rrr(theta1, theta2, theta3, l1, l2, l3, alpha1, alpha2, alpha3)
    error = np.sqrt((x_end - x_target)**2 + (y_end - y_target)**2 + (z_end - z_target)**2)
    if error < 1e-6:  # Tolerance for numerical precision
        print("Solution is valid! End-effector matches the target.")
    else:
        print(f"Solution is not exact. Error: {error:.6f}")


def plot_robot(theta1, theta2, theta3, l1, l2, l3, alpha1, alpha2, alpha3, x_target, y_target, z_target):
    """
    Plots the robot configuration given joint angles, link lengths, and twists.
    """
    # Base position
    x0, y0, z0 = 0, 0, 0

    # Link 1 end
    x1, y1, z1 = 0, 0, l1

    # Link 2 end
    x2 = x1 + l2 * np.cos(theta1) * np.cos(theta2 + alpha2)
    y2 = y1 + l2 * np.sin(theta1) * np.cos(theta2 + alpha2)
    z2 = z1 + l2 * np.sin(theta2 + alpha2)

    # Link 3 end (End-effector)
    x3 = x2 + l3 * np.cos(theta1) * np.cos(theta2 + theta3 + alpha3)
    y3 = y2 + l3 * np.sin(theta1) * np.cos(theta2 + theta3 + alpha3)
    z3 = z2 + l3 * np.sin(theta2 + theta3 + alpha3)

    # Plot the robot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot links
    ax.plot([x0, x1], [y0, y1], [z0, z1], 'ro-', label="Link 1 (Base to Joint 1)")
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'go-', label="Link 2 (Joint 1 to Joint 2)")
    ax.plot([x2, x3], [y2, y3], [z2, z3], 'bo-', label="Link 3 (Joint 2 to End-effector)")

    # Plot target and end-effector
    ax.scatter([x_target], [y_target], [z_target], color="orange", s=50, label="Target")
    ax.scatter([x3], [y3], [z3], color="purple", s=50, label="End-effector")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Compute the ranges
    x_range = [min(x0, x1, x2, x3, x_target), max(x0, x1, x2, x3, x_target)]
    y_range = [min(y0, y1, y2, y3, y_target), max(y0, y1, y2, y3, y_target)]
    z_range = [min(z0, z1, z2, z3, z_target), max(z0, z1, z2, z3, z_target)]
    max_range = max(np.ptp(x_range), np.ptp(y_range), np.ptp(z_range)) / 2.0

    # Center points
    x_mid = (x_range[0] + x_range[1]) / 2.0
    y_mid = (y_range[0] + y_range[1]) / 2.0
    z_mid = (z_range[0] + z_range[1]) / 2.0

    # Set equal axis limits
    ax.set_xlim([x_mid - max_range, x_mid + max_range])
    ax.set_ylim([y_mid - max_range, y_mid + max_range])
    ax.set_zlim([z_mid - max_range, z_mid + max_range])

    # Enforce equal scaling visually
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes

    ax.legend()
    ax.set_title("3-DOF RRR Robot Configuration with Twists")
    plt.show()



# Example usage
if __name__ == "__main__":
    # Link lengths
    l1, l2, l3 = 1, 1.0, 0.5

    # Joint twists
    alpha1 = np.radians(-90)
    alpha2 = np.radians(0)
    alpha3 = np.radians(0)

    # Target position
    x_target, y_target, z_target = 1., -0.5, 1.5

    try:
        print("Target:",x_target,y_target,z_target)
        # Solve IK
        theta1, theta2, theta3 = inverse_kinematics_3dof_rrr(x_target, y_target, z_target, l1, l2, l3, alpha1, alpha2, alpha3)

        # Validate
        x,y,z = forward_kinematics_3dof_rrr(theta1, theta2, theta3, l1, l2, l3, alpha1, alpha2, alpha3)
        print("Actual:",x,y,z)
        validate_solution(x_target, y_target, z_target, theta1, theta2, theta3, l1, l2, l3, alpha1, alpha2, alpha3)

        # Plot
        plot_robot(theta1, theta2, theta3, l1, l2, l3, alpha1, alpha2, alpha3, x, y, z)
        
    except ValueError as e:
        print(e)
