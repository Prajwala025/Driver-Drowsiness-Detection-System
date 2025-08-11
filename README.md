# Driver-Drowsiness-Detection-System
Driver Drowsiness Detection Alert System Using Python
A tool designed to detect driver drowsiness through face recognition and alert the driver with voice commands. This project leverages Python and machine learning concepts to ensure driver safety by monitoring yawning and eye focus.
## Overview
This project uses several powerful libraries and frameworks:
- **dlib**: A machine learning library for face recognition and landmark prediction.
- **imutils**: A collection of convenience functions to work with images and video frames.
- **scipy**: Used to calculate yawning and eye focus percentages.
- **OpenCV**: A comprehensive library for image and video processing.
- **argparse**: Provides a command-line interface for running the project with specific parameters.

Hardware Requirements
- A camera with at least 2.0 MP resolution.
- An audio speaker for alerts.
- A Raspberry Pi or a computer system to connect and process the frames using Python.
Project Functionality
The camera captures 10 frames per second. Each frame is checked against the defined thresholds for yawning and eye focus. Alerts are triggered via the speaker:
- **Eye Alert**: "Open your eyes, sir."
- **Yawn Alert**: "Take some fresh air, sir."
Press `q` to quit the application.
Challenge
The most challenging part was installing the `dlib` library, as many Python versions do not support it directly. After extensive research, the compatible versions were identified as Python 3.8.5 and `dlib` 19.19.0. Although an XML file can be used for face detection, it is less accurate than `dlib`.
Installation Guide
-Prerequisites
- **Python 3.8.5**: Ensure that Python 3.8.5 is installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/release/python-385/).
- **Microsoft Visual Studio C++ Build Tools**: Required for compiling some of the libraries.
    1. Download the [Microsoft Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    2. During installation, ensure you select the **C++ build tools** and the **Windows 10 SDK**.
*Create a Virtual Environment
In command prompt type  python -m venv venv
*Activate the Virtual Environment
 On Windows  venv\Scripts\activate
*Install Required Libraries
Install CMake pip install cmake
*Install Other Dependencies
 pip install -r requirements.txt
*Run the Application
   python main.py
