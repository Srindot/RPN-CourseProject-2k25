# Usage Instructions

### Prerequisites

- Docker Installation
  ```bash
  # Install Docker using convenience script
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh ./get-docker.sh

  # Post-install configuration
  sudo groupadd docker
  sudo usermod -aG docker $USER
  sudo systemctl enable docker.service
  sudo systemctl enable containerd.service

  # Verify installation
  sudo systemctl is-enabled docker
  ```

 **Reboot before proceeding further**

**GHCR Authentication**

```bash
  echo "<YOUR_GITHUB_PAT>" | docker login ghcr.io -u <YOUR_GITHUB_USERID> --password-stdin
```

##### Prerequisites

- VSCode
- Remote Development Extension by Microsoft (Inside VSCode)

##### Setup Process

- Create a folder for P3DX development
  ```bash
  mkdir P3DX && cd P3DX
  # Clone the repo 
  git clone https://github.com/rtarun1/P3DX-Docker.git .
  # Open VSCode 
  code .
  ```
- To enter the container
  - Open Command Pallete with `Ctrl+Shift+P`
  - Select **Dev Containers: Reopen in Container**
  - Use `Build WS` button to build workspace

## Start up the ROS Master Node

``roscore``

## Setup connection with the robot

**Make sure to source the workspace on each terminal.**

``source devel/setup.bash``

``sudo chmod +777 /dev/ttyUSB0``

``rosrun rosaria RosAria``

## Setting up the Camera

``roslaunch realsense2_camera rs_camera.launch align_depth:=true``

To get pointclouds from the RGBD images:

``roslaunch realsense2_camera rs_camera.launch align_depth:=true enable_pointcloud:=true``

## Setup the conversion of RGBD images to Laser Scans

``roslaunch depthimage_to_laserscan launchfile_sample.launch``

## Setup the navigation node

``roslaunch turtlebot3_navigation turtlebot3_navigation.launch``

## Setting up Parser Based Navigation

``rosrun turtlebot3_navigation talker``

``rosrun turtlebot3_navigation Voice_NLP``

## Start up the ROS

1. Launch
   ```
   sudo chmod +777 /dev/ttyUSB0
   roslaunch husky_base base.launch 
   ```

   - Optionally, you can plug a joystick and teleop the robot.

## Docker

- To permanently add any ROS APT packages, list them in the rosPkgs.list file, then rebuild the Docker image using:

  ```
  docker build -t ghcr.io/rtarun1/husky_base -f .devcontainer/Dockerfile .devcontainer
  ```
- Always run ``sudo apt update`` inside the container before installing any additional packages.
- For Docker-related questions or issues, feel free to open an issue on the [DockerForROS2Development](https://github.com/soham2560/DockerForROS2Development.git)

### For Arch users :(

```bash
echo "127.0.0.1 stimpy" | sudo tee -a /etc/hosts

```

### Credits

- This Docker setup was adapted from [Soham&#39;s repository](https://github.com/soham2560/DockerForROS2Development.git).
- If you use this repository for your project or publication, please consider citing or acknowledging the contributors accordingly.

# Running RTABMap with RRC P3DX

## Initial setup and teleop

- Go to Laksh’s P3DX repo and git clone it into a separate workspace https://github.com/laksh-nanwani/P3DX and follow the instructions there to setup the port and catkin_make the repo.
- Next, after turning the robot power ON, run the `rosrun rosaria RosAria` command to establish connection with the robot, which can be conformed by a beep from the robot.
- Next, to teleop the robot, run `f`  which will give you the keyboard control over the robot. W -forwards, A- left, D- right, S- stop, X- backwards.

## RTABMap Mapping and localization

- After connecting the camera through the type C cable, you can run the `roslaunch realsense2_camera rs_camera.launch align_depth:=true` , which will publish the camera topics.
- Next for mapping, run the

  `roslaunch rtabmap_launch rtabmap.launch \ rtabmap_args:="--delete_db_on_start" \ depth_topic:=/camera/aligned_depth_to_color/image_raw \ rgb_topic:=/camera/color/image_raw \ camera_info_topic:=/camera/color/camera_info \ approx_sync:=false`
- In case if you want to view the recorded map, run `rtabmap-databaseViewer ~/.ros/rtabmap.db`
- Finally for localizaiton, run `roslaunch rtabmap_launch rtabmap.launch localization:=true`
