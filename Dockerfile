FROM ros:humble

# Setup dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-opencv \
    libopencv-dev \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install opencv-python


WORKDIR /workspace

