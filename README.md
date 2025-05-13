# DETECTION-OF-FACE-SWAPPED-DEEP-FAKE-VIDEOS

# Deepfake Detector Web Application

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2+-green.svg)](https://www.djangoproject.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-orange.svg)](https://pytorch.org/)
[![Firebase](https://img.shields.io/badge/Firebase-Cloud-yellow.svg)](https://firebase.google.com/)

## Overview

This project is a web application designed to detect deepfake videos. It allows users to upload video files, which are then analyzed by a deep learning model to determine if they have been digitally manipulated. The system leverages a combination of computer vision techniques, a deep neural network architecture (ResNeXt and LSTM), and a Django-based web interface with Firebase for user authentication.

## Key Features

* **Deepfake Detection:** Utilizes a trained deep learning model to classify uploaded videos as either "REAL" or "FAKE," with a confidence score. The model is specifically trained to identify facial manipulations common in deepfakes.
* **User-Friendly Web Interface:** Provides an intuitive web interface built with Django, allowing users to easily upload videos and view the analysis results.
* **Firebase Authentication:** Secure user registration and login system powered by Firebase Authentication.
* **Account Management:** Users can create accounts, log in, and log out.
* **Result Visualization:** Displays the prediction result along with a confidence breakdown in the form of a pie chart.
* **Frame Analysis:** Provides insights by displaying processed frames and cropped face regions from the analyzed video.
* **Informational Pages:** Includes "About Us," "Contact," and "Legal" pages for additional information.

## Technology Stack

* **Backend:**
    * Python (version 3.6 or higher)
    * Django (version 4.2 or higher) - Web framework
    * PyTorch - Deep learning framework for the detection model
    * Firebase Admin SDK - For interacting with Firebase services
    * OpenCV (`cv2`) - For video processing and frame extraction
    * face\_recognition - For face detection in video frames
    * Pillow (PIL) - For image manipulation
    * NumPy - For numerical operations
    * Matplotlib and Seaborn - For visualization (generating the confidence pie chart during analysis)
* **Frontend:**
    * HTML
    * CSS
    * JavaScript (as rendered by Django templates)
* **Authentication & Data Storage:**
    * Firebase Authentication - For user account management
    * Firestore - For storing user data, including password hashes

## Setup and Running Instructions

Follow these steps to set up and run the Deepfake Detector application on your local machine.

### Prerequisites

* **Python 3:** Ensure you have Python 3 installed (version 3.6 or higher recommended). Download from [python.org](https://www.python.org/). Make sure to add Python to your system's PATH during installation.
* **pip:** Python's package installer, usually included with Python 3.
* **Firebase Service Account Key File:** You should have received a JSON file containing the credentials for the pre-configured Firebase project. Place this file in your project directory.

### Setup Steps

1.  **Unzip the Project:** Extract the contents of the provided zip file to a folder on your computer. Let's assume the main project folder is named `deepfake_detector`.

2.  **Open a Terminal or Command Prompt:** Navigate to the extracted project folder in your terminal.

    ```bash
    cd /path/to/your/deepfake_detector_folder
    ```

3.  **Install Project Dependencies:** Install all the required Python libraries using pip.

    ```bash
    python3 -m pip install Django firebase-admin Pillow opencv-python face_recognition matplotlib torch torchvision
    ```

    **Note on PyTorch:** The installation of PyTorch might require specific configurations based on your operating system and whether you have a GPU. Refer to the official PyTorch installation guide ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) if you encounter issues. For CPU-only usage, the command above should suffice.

4.  **Configure Firebase Service Account Key Path:**
    * Open the `deepfake_detector/settings.py` file.
    * Locate the `FIREBASE_SERVICE_ACCOUNT_KEY_PATH` setting.
    * Update the path to correctly point to the JSON service account key file you received. For example:

        ```python
        FIREBASE_SERVICE_ACCOUNT_KEY_PATH = os.path.join(BASE_DIR, 'your_firebase_key.json')
        ```

        Replace `'your_firebase_key.json'` with the actual name of your JSON file.

5.  **Apply Migrations:** Django uses migrations to manage database schema. Even though this project primarily uses Firestore, Django might require some initial setup.

    ```bash
    python3 manage.py migrate
    ```

6.  **Run the Development Server:** Start the Django development server to run the application locally.

    ```bash
    python3 manage.py runserver
    ```

    You might see a message indicating the server has started at `http://127.0.0.1:8000/`.

7.  **Access the Application:** Open your web browser and navigate to `http://127.0.0.1:8000/`.

### Using the Application

1.  **Signup/Login:**
    * If you are a new user, click on the "Signup" link in the header to create an account. This process uses Firebase Authentication.
    * If you already have an account, click on the "Login" link and enter your credentials.
2.  **Upload Video:**
    * After logging in, you will be directed to the "Upload" page.
    * Click on the "Choose File" button to select a video file from your computer.
    * Click the "Upload" button to submit the video for analysis.
3.  **View Results:**
    * Once the video is processed, you will be redirected to the "Results" page.
    * The page will display the detection outcome ("REAL" or "FAKE") along with a confidence score.
    * A pie chart visualizing the confidence breakdown will also be shown.
    * Below the results, you may see a selection of processed frames and cropped face regions from the video that were analyzed by the model.
4.  **Navigation:**
    * Use the links in the header to navigate between the "Upload," "About Us," "Contact," and "Legal" pages.
    * Click the "Logout" link in the header when you are done.

### Troubleshooting

* **`NameError` or `ModuleNotFoundError`:** Ensure you have installed all the required dependencies using `python3 -m pip install`.
* **Firebase Errors (e.g., `CONFIGURATION_NOT_FOUND`, "404 Database does not exist"):** These errors might indicate an issue with the Firebase project configuration. Double-check that Email/Password Authentication and Firestore are enabled in the Firebase project associated with your service account.
* **Error: "That port is already in use."** If you see this error when running the server, it means another application is using port 8000. Try running the server on a different port (e.g., `python3 manage.py runserver 8001`).
* **Issues with ML Model (if applicable):** If you encounter errors related to PyTorch or the model, ensure you have installed the correct version of `torch` and `torchvision` for your system (CPU/GPU). Refer to the official PyTorch installation guide.
* **Issues with the service account key:** Double-check that the `FIREBASE_SERVICE_ACCOUNT_KEY_PATH` in `settings.py` is correctly pointing to the JSON file you received.

## Contributing

If you would like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them.
4.  Push your changes to your fork.
5.  Submit a pull request.

## License

[Specify your project license here, e.g., MIT License]

## Contact

[Your Name/Organization]
[Your Email/Website]
