# 🤖 RRRP Manipulator Simulation

**Author:** Aman Mishra  
**Module:** PDE 4431 Robot Manipulation  
**Date:** December 2025

---

## Introduction

This project simulates a **4-DOF Revolute-Revolute-Revolute-Prismatic (RRRP) manipulator** for complex pick-and-place operations. The simulation demonstrates the robot's ability to transfer payloads from a table to three vertical shelf tiers (**A**, **B**, and **C**) at varying heights.

The kinematic model is established using the **Denavit-Hartenberg (DH) convention**. The configuration involves:

-   SCARA-like base and shoulder
-   Pitching elbow
-   Downward-extending prismatic gripper

---

## Denavit-Hartenberg (DH) Table

Coordinate frames are assigned so the prismatic extension ($d_4$) occurs along the World $-Z$ axis.

| Link (i)      | $a_i$ (Length) | $\alpha_i$ (Twist) | $d_i$ (Offset) | $\theta_i$ (Angle) |
| ------------- | :------------: | :----------------: | :------------: | :----------------: |
| 1 (Base)      |     $2.0$      |     $0^\circ$      |     $2.0$      |    $\theta_1^*$    |
| 2 (Shoulder)  |     $2.0$      |    $-90^\circ$     |      $0$       |    $\theta_2^*$    |
| 3 (Elbow)     |     $2.0$      |    $-90^\circ$     |      $0$       |    $\theta_3^*$    |
| 4 (Prismatic) |      $0$       |     $0^\circ$      |    $d_4^*$     |     $0^\circ$      |

_\* Asterisk indicates variable joint parameters solved by the Inverse Kinematics algorithm._

---

## Demonstration Video

[YouTube Link](#) <!-- Replace # with actual link when available -->

---

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/MrRox1337/manipulator-kinematics.git
    ```
2. Navigate to the project directory:
    ```bash
    cd manipulator-kinematics
    ```
3. Run the main simulation script:
    ```bash
    python dh_transform_fwd_inv_rrrp.py
    ```

---

## Codebase Structure

```
manipulator-kinematics/
├── main/
│   └── dh_transform_fwd_inv_rrrp.py   # Main simulation and kinematics
├── test/
│   ├── dh_transform_fwd.py            # Forward kinematics (test)
│   ├── dh_transform_fwd_2dsweep.py    # 2D sweep simulation (test)
│   └── kinematics.py                  # Kinematic helper functions (test)
├── README.md
```

---

## License

This project is for academic use in PDE 4431 Robot Manipulation.
