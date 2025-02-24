Gesture-Controlled Sound Synthesizer

Overview

sound.py is an interactive Python script that generates and manipulates sound based on hand gestures. Using computer vision and audio processing, it detects hand movements to control sound properties like frequency, volume, and beats per minute (BPM).

Features

Hand Gesture Recognition: Uses OpenCV and MediaPipe for tracking hand movements.

Real-time Sound Generation: Utilizes sounddevice to produce dynamic sound output.

8D Stereo Effect: Modulates stereo panning for an immersive experience.

Multi-threading: Ensures smooth performance by handling audio and vision processing separately.

Adjustable Sound Parameters:

Frequency: Controlled by right-hand gestures (200-2000 Hz).

Volume: Adjusted based on hand distance (0.0 - 1.0 amplitude).

BPM: Modified by left-hand gestures (60-180 BPM).

Installation

Ensure you have Python installed, then install the dependencies:

pip install -r requirements.txt

Usage

Run the script to start the gesture-controlled synthesizer:

python sound.py

Dependencies

opencv-python

mediapipe

numpy

sounddevice

Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

License

This project is licensed under the MIT License.
