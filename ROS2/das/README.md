# Tu run ROS2 do the following:

To build:
```bash
colcon build
source install/setup.bash
```

To run the aggregative optimization:
```bash
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
ros2 launch launch/launch.py
```

To run the visualization you need 2 terminals:
1. Terminal 1:
```bash
ros2 run das rviz  
```
2. Terminal 2:
```bash
rviz2 -d default.rviz
```
