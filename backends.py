# detector/backends.py

# Keep existing imports
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model

import firebase_admin
from firebase_admin import auth, credentials
import os
from django.conf import settings

# Import check_password
from django.contrib.auth.hashers import check_password

# Import firestore
from firebase_admin import firestore


UserModel = get_user_model()

class FirestoreBackend(BaseBackend):
    """
    Custom authentication backend to authenticate users against Firestore
    using a password hash stored in Firestore and linked by Firebase UID.
    """
    def authenticate(self, request, username=None, password=None, **kwargs):
        """
        Authenticates a user by getting the Firebase UID from email via Firebase Auth,
        then verifying the password against a hash stored in Firestore.
        """
        if not username or not password:
            return None

        try:
            # Step 1: Get the Firebase User record by email to obtain the UID
            # This verifies the email exists in Firebase Auth but not the password
            user_record = auth.get_user_by_email(username)
            uid = user_record.uid # Get the Firebase UID

            # Step 2: Retrieve the user document from Firestore using the UID
            db = firestore.client()
            user_ref = db.collection('users').document(uid)
            user_doc = user_ref.get()

            if not user_doc.exists:
                print(f"Firestore document not found for UID: {uid}")
                return None # User exists in Firebase Auth but not in our Firestore collection

            user_data = user_doc.to_dict()
            stored_password_hash = user_data.get('password_hash')

            if not stored_password_hash:
                 print(f"Password hash not found in Firestore for UID: {uid}")
                 return None # Password hash is missing in Firestore

            # Step 3: Verify the provided password against the hash from Firestore
            if check_password(password, stored_password_hash):
                # Password is correct, get or create the Django User
                user, created = UserModel.objects.get_or_create(
                    username=uid, # Use Firebase UID as Django username
                    defaults={'email': user_record.email}
                )
                print(f"User authenticated successfully: {user.username}")
                return user # Authentication successful
            else:
                # Password is incorrect
                print(f"Password verification failed for user: {username}")
                return None # Authentication failed (wrong password)

        except firebase_admin.auth.UserNotFoundError:
            # User does not exist in Firebase Authentication by email
            print(f"User with email {username} not found in Firebase Auth.")
            return None
        except Exception as e:
            print(f"Error during Firestore/password authentication: {e}")
            # In a real app, log the full traceback: traceback.print_exc()
            return None


    def get_user(self, user_id):
        """
        Retrieves a Django User object based on the user_id (which is the Django User's primary key).
        """
        try:
            # Retrieve the Django User by its primary key (user_id)
            # Django stores the user's primary key (pk) in the session.
            return UserModel.objects.get(pk=user_id)
        except UserModel.DoesNotExist:
            print(f"Django User with pk {user_id} does not exist.")
            return None
        except Exception as e:
            print(f"Error retrieving Django user by pk: {e}")
            return None