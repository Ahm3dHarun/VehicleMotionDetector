# Motion Detection Project with Kalman Filters

This project aims to detect and track moving objects in a video using motion detection and Kalman filters. The system maintains a list of object candidates detected over time and tracks each object using Kalman filters. Below, we'll explain the different classes involved and how to run the project.

## Project Structure

The project consists of two main classes:

1. **BasicKalmanFilter**:

   - This class implements a basic Kalman filter for tracking individual objects.
   - It provides methods to initialize the filter, predict the next state, update the filter with new measurements, and retrieve the current state and position history.

2. **MotionDetection**:
   - This class manages the motion detection and tracking logic.
   - It initializes the background subtractor, detects moving objects, updates existing trackers, removes inactive trackers, and initializes new trackers as needed.
   - It provides methods to update the detection and tracking process and retrieve the current positions of tracked objects and their position histories.

## Running the Project

To run the project, follow these steps:

1. **Install Required Libraries**:

   - Make sure you have Python installed on your system.
   - Install the necessary libraries by running:
     ```
     pip install opencv-python numpy filterpy
     ```

2. **Prepare Your Video**:

   - Replace `'video.mp4'` in the code with the path to your video file.
   - Ensure that the video contains moving objects that you want to detect and track.

3. **Run the Script**:

   - Save the provided Python script (`motion_detection.py`) on your local machine.
   - Open a terminal or command prompt and navigate to the directory containing the script.
   - Run the script using the following command:
     ```
     python motion_detection.py
     ```

4. **View the Output**:
   - The script will open a window displaying the video with detected and tracked objects.
   - Press the 'Esc' key to close the window and stop the script.

## Notes:

- Ensure that your video file is accessible and properly formatted.
- You can adjust the hyperparameters in the code to fine-tune the motion detection and tracking performance.
- Feel free to experiment with different videos and settings to observe how the system detects and tracks moving objects.
