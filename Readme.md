# Falcon AMR A* Planner

## Overview

This project runs an A* path planner for the Falcon AMR path planning simulation using ROS 2. The planner computes a collision free path on a 2D map, accounts for robot radius and clearance, and sends velocity commands to the Falcon robot through `cmd_vel`.

The workflow uses two terminals:

1. Publish a static transform between the Falcon map frame and robot frame.
2. Launch the Falcon simulator and A* controller with the desired start, goal, robot size, wheel dimensions, and RPM values.

Video demonstration:

```text
https://youtu.be/OYTKofNbG3M
```

---

## File Structure

Only one main Python file is used for the planner logic:

```bash
astar_falcon_planner/
└── astar_planner.py
```

Supporting ROS files:

```bash
astar_falcon_planner/
├── ros_falcon_astar.launch.py
├── package.xml
└── AMRPathPlanning.usda
```

---

## Description

The planner uses a 400 cm × 200 cm map, which corresponds to the 4 m × 2 m Falcon world. Falcon launch inputs are given in meters, then converted to centimeters before being passed into the A* planner.

The obstacle map is hard coded inside `astar_planner.py` using half-plane equations. Obstacle inflation is based on:

```text
5cm + clearance
```

The A* planner expands differential drive motion primitives using RPM combinations and returns a path as incremental motion commands:

```text
[dx, dy, dtheta]
```

where `dx` and `dy` are in centimeters and `dtheta` is in radians.

---

## Install Dependencies

Install OpenCV:

```bash
pip install opencv-python
```

---

## How to Run

### Terminal 1: Publish Static Transform

Run:

```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map IMUSensor_BP_C_0
```

Keep this terminal running.

### Terminal 2: Launch Falcon + A*

Run:

```bash
ros2 launch astar_falcon_planner ros_falcon_astar.launch.py \
start_position:=[.70,1.55,0.0] \
end_position:=[1.7,1.7,0.0] \
robot_radius:=0.14350 \
clearance:=0.000 \
delta_time:=1.0 \
wheel_radius:=0.033 \
wheel_distance:=0.287 \
rpms:=[70.0,100.0]
```

---

## Updating the Start Position in Falcon

Whenever the start position is changed in the launch command, also update the robot start pose in:

```bash
AMRPathPlanning.usda
```

Update line 38:

```text
double3 xformOp:translate = (1670.0, 755.0, 0.0)
```

This keeps the Falcon robot spawn location consistent with the A* planner start position.

---

## Launch Parameters

| Parameter | Description | Unit |
|---|---|---|
| `start_position` | Robot start pose `[x,y,theta]` | meters, degrees |
| `end_position` | Goal pose `[x,y,theta]` | meters, degrees |
| `robot_radius` | Robot radius | meters |
| `clearance` | Extra safety distance from obstacles | meters |
| `delta_time` | Time step used for command execution | seconds |
| `wheel_radius` | Wheel radius | meters |
| `wheel_distance` | Distance between wheels | meters |
| `rpms` | Wheel RPM values used by A* actions | RPM |

---

## Notes

- Falcon uses meters.
- `astar_planner.py` uses centimeters internally.
- The ROS node converts launch parameters from meters to centimeters before calling `plan_path()`.
- The obstacle map itself is already defined in centimeters.
- `SCALE = 1` is only for visualization and does not affect the planner geometry.

