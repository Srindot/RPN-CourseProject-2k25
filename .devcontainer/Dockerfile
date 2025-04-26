FROM ghcr.io/rtarun1/p3dx_base:v1.0

# Install dependencies and ROS Noetic key
RUN sudo apt update && sudo apt install -y curl && \
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS2 Foxy repository
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update & upgrade packages
RUN sudo apt update -y && sudo apt upgrade -y

# Install ROS2 Foxy and necessary tools
RUN sudo apt install -y ros-foxy-desktop python3-argcomplete ros-dev-tools

# Installing Navigation2 (Nav2) for foxy 
RUN sudo apt update && \
    sudo apt install -y ros-foxy-navigation2 ros-foxy-nav2-bringup

# installing the Turtle Bot gazebo for navigation 
# RUN sudo apt install ros-foxy-turtlebot3-gazebo
# install ros2 -> ros1 bridge 
RUN sudo apt install ros-foxy-ros1-bridge

CMD ["/bin/bash"]
