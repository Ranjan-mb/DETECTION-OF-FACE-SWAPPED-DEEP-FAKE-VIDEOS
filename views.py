from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import VideoUploadForm
from .model_integration.predict_utils import get_prediction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
import cv2
import face_recognition
from PIL import Image as pImage
import time
from django.conf import settings
from torchvision import transforms
import numpy as np
from django.shortcuts import redirect # For redirects
from firebase_admin import auth
from firebase_admin.auth import EmailAlreadyExistsError # Import specific Firebase Auth error
from django.contrib.auth import authenticate, login, logout # Import authenticate, login, and logout
from django.conf import settings # Import settings to use LOGIN_REDIRECT_URL
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password, check_password # <-- This line is needed
from firebase_admin import firestore

# Assuming you added LoginForm to forms.py, import it:
from .forms import VideoUploadForm, SignupForm, LoginForm # Add LoginForm here


MODEL_PATH = './detector/model_integration/model_97_acc_100_frames_FF_data.pt' # Adjust if your path is different
IM_SIZE = 112
PADDING = 40
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
SEQUENCE_LENGTH = 60 # You might want to get this from the form as well
def cleanup_uploaded_images():
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

@login_required
def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            cleanup_uploaded_images()

            video_file = request.FILES['video_file']
            saved_video_path = './temp_video.mp4' # Consider a more robust temporary file handling
            with open(saved_video_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
            print(f"File uploaded: {video_file.name}")
            print(f"File size: {video_file.size}")

            # --- Start timing for processing ---
            start_time = time.time()
            # --- End timing ---

            prediction_value, confidence = get_prediction(saved_video_path, MODEL_PATH)

            # --- Calculate processing time ---
            end_time = time.time()
            processing_time = end_time - start_time
            # --- End calculation ---


            if prediction_value == -1:
                # Clean up the temporary video file in case of an error
                if os.path.exists(saved_video_path):
                     os.remove(saved_video_path)
                return render(request, 'detector/result.html', {'error': 'Could not process the video.'})

            prediction_label = "REAL" if prediction_value == 1 else "FAKE"

            # Create a themed pie chart
            labels = ['REAL', 'FAKE']
            values = [confidence, 100 - confidence] if prediction_label == "REAL" else [100 - confidence, confidence]
            colors = ['#00ff99', '#ff3b3f']  # neon green for REAL, neon red for FAKE

            # Dark mode theme
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(4, 4), facecolor='#1e1e2f')
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'color': 'white', 'fontsize': 12},
                wedgeprops={'edgecolor': '#1e1e2f'}
            )
            plt.setp(autotexts, size=13, weight="bold")
            ax.set_title("Confidence Breakdown", color='white')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
            buffer.seek(0)
            image_png = buffer.getvalue()
            chart = base64.b64encode(image_png).decode('utf-8')
            plt.close()

            # Process frames and extract faces
            preprocessed_images = []
            faces_cropped_images = []
            cap = cap = cv2.VideoCapture(saved_video_path)

            if not cap.isOpened():
                 # Clean up the temporary video file in case of an error
                if os.path.exists(saved_video_path):
                     os.remove(saved_video_path)
                return render(request, 'detector/result.html', {'error': 'Could not open the video file.'})
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            cap.release()

            # --- Calculate frames analyzed ---
            frames_analyzed = len(preprocessed_images) # Count how many preprocessed frames were saved
            # --- End calculation ---

            video_file_name_only = "uploaded_video_" + str(int(time.time())) # Create a unique name

            # Set upload_dir directly to settings.MEDIA_ROOT
            upload_dir = settings.MEDIA_ROOT
            # Ensure the upload directory exists
            os.makedirs(upload_dir, exist_ok=True)


            for i in range(SEQUENCE_LENGTH): # The loop limit determines how many frames are attempted to be processed
                if i >= len(frames):
                    break
                frame = frames[i]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                # Save preprocessed image
                image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
                image_path = os.path.join(upload_dir, image_name)
                # Ensure the file is saved correctly
                try:
                    pImage.fromarray(rgb_frame).save(image_path)
                    preprocessed_images.append(image_name)
                except Exception as e:
                    print(f"Error saving preprocessed image {image_name}: {e}")
                    # Handle this error appropriately


                # Face detection and cropping
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    frame_face = frame[max(0, top - PADDING):min(frame.shape[0], bottom + PADDING),
                                       max(0, left - PADDING):min(frame.shape[1], right + PADDING)]
                    if frame_face.size > 0:
                        face_image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
                        face_image_path = os.path.join(upload_dir, face_image_name)
                        # Ensure the file is saved correctly
                        try:
                            pImage.fromarray(cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)).save(face_image_path)
                            faces_cropped_images.append(face_image_name)
                        except Exception as e:
                             print(f"Error saving cropped face image {face_image_name}: {e}")
                             # Handle this error appropriately

            print("Preprocessed Images:", preprocessed_images)
            print("Cropped Faces Images:", faces_cropped_images)

            # Clean up the temporary video file after processing
            if os.path.exists(saved_video_path):
                 os.remove(saved_video_path)

            return render(request, 'detector/result.html', {
                'prediction': prediction_label,
                'confidence': f'{confidence:.2f}',
                'chart': chart,
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'MEDIA_URL': settings.MEDIA_URL,
                'frames_analyzed': frames_analyzed, # Pass frames analyzed
                'processing_time': f'{processing_time:.2f}s', # Pass formatted processing time
            })

    else:
        form = VideoUploadForm()

    # Ensure temporary video is cleaned up even if form is invalid or not POST
    # This might need more robust handling if temporary file saving fails initially
    # if request.method == 'POST' and os.path.exists(saved_video_path):
    #      os.remove(saved_video_path) # This would require saved_video_path to be in scope


    return render(request, 'detector/upload.html', {'form': form})

def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password'] # Get the raw password

            try:
                # Create user in Firebase Authentication (this is mainly to get a stable UID)
                # We are *not* relying on Firebase Auth's password management for login verification
                # in this specific backend approach.
                user_record = auth.create_user(
                    email=email,
                    # Setting a dummy password here if create_user requires one,
                    # or relying on Firebase Auth's internal hashing if you pass the real one.
                    # If you pass the real password here, Firebase Auth will hash it internally,
                    # but our backend will verify against the hash stored in Firestore.
                    # For this approach, let's create the user with the password
                    # so a complete Firebase Auth record exists, but our backend
                    # will use the Firestore hash for verification.
                    password=password, # Pass the password to create the Firebase Auth user
                    # display_name='New User' # Optional
                )
                print(f'Successfully created new Firebase Auth user record: {user_record.uid}')

                # Hash the password using Django's utility
                hashed_password = make_password(password)

                # Save user data and the hashed password to Firestore
                db = firestore.client()
                user_ref = db.collection('users').document(user_record.uid) # Use Firebase UID as Firestore doc ID
                user_ref.set({
                    'email': email,
                    'password_hash': hashed_password, # Store the hashed password
                    'created_at': firestore.SERVER_TIMESTAMP,
                    # Add other profile fields as needed
                })
                print(f'Saved user data and password hash to Firestore for UID: {user_record.uid}')

                # Redirect to the login page after successful signup
                return redirect('login')

            except EmailAlreadyExistsError:
                form.add_error('email', 'This email address is already in use.')
            except Exception as e:
                # Handle other potential errors during Firebase user creation or Firestore saving
                print(f"Error during signup: {e}")
                # In a real app, log the full traceback: traceback.print_exc()
                form.add_error(None, 'An unexpected error occurred during signup. Please try again.')

    else:
        form = SignupForm()

    # Render the signup template
    return render(request, 'detector/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']

            # Use Django's authenticate function
            # This will use your AUTHENTICATION_BACKENDS, including FirestoreBackend
            user = authenticate(request, username=email, password=password) # Pass email as username

            if user is not None:
                # User is authenticated
                login(request, user) # Log the user in, establishing a session
                print(f"User logged in: {user.username}")
                # Redirect to the page after login (defined in settings.LOGIN_REDIRECT_URL)
                return redirect(settings.LOGIN_REDIRECT_URL)
            else:
                # Authentication failed
                form.add_error(None, 'Invalid email or password.')

    else:
        form = LoginForm()

    # Render the login template
    return render(request, 'detector/login.html', {'form': form})

def logout_view(request):
    logout(request) # Use Django's built-in logout function
    print("User logged out.")
    # Redirect to the page after logging out (defined in settings.LOGOUT_REDIRECT_URL)
    return redirect(settings.LOGOUT_REDIRECT_URL)
def about_view(request):
    """
    Renders the About Us page.
    """
    return render(request, 'detector/about.html', {}) # No extra context needed for now


def contact_view(request):
    """
    Renders the Contact page.
    """
    return render(request, 'detector/contact.html', {}) # No extra context needed for now


def legal_view(request):
    """
    Renders the Legal page (Terms, Privacy, etc.).
    """
    return render(request, 'detector/legal.html', {}) # No extra context needed for now
