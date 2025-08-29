🖌️ Air Draw - Gesture Controlled Drawing

    Air Draw is a computer vision project that allows you to draw on a virtual canvas using just your hand gestures 🖐️.Using OpenCV and MediaPipe, the project detects your hand landmarks in real-time and lets you draw lines by simply raising your index finger.

✨ Features

    🎥 Real-time Hand Tracking using MediaPipe
    🖐️ Gesture Based Drawing – draw only when the index finger is up
    📏 Smooth & Stable Lines using Exponential Moving Average (EMA)
    🧹 Canvas Reset – press C to clear the canvas
    ❌ Exit Anytime – press Q to quit the program
    🎨 Easily customizable drawing color and thickness

⚙️ Technologies Used

    1. Python 3.x 🐍
    2. OpenCV (for image processing and display)
    3. MediaPipe (for hand landmark detection)
    4. NumPy (for handling canvas operations)\

🚀 How to Run

    1. Clone the repository
        git clone https://github.com/DeepakdevilB/Air_Draw.git
        cd Air_Draw

    2. Install Dependencies
        pip install opencv-python mediapipe numpy

    3. Run The Script 
        python air_draw.py


🎮 Controls

    Raise index finger 👉 to start drawing
    Press C 🧹 to clear canvas
    Press Q ❌ to quit
