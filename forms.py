from django import forms
from django.core.exceptions import ValidationError


class VideoUploadForm(forms.Form):
    video_file = forms.FileField(label='Select a video file')
    sequence_length = forms.IntegerField(label='Sequence Length', min_value=1, initial=10)

class SignupForm(forms.Form):
    email = forms.EmailField(
        label="Email",
        max_length=254,
        widget=forms.EmailInput(attrs={'placeholder': 'Enter your email'})
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={'placeholder': 'Enter your password'})
    )
    password_confirm = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={'placeholder': 'Confirm your password'})
    )

    def clean(self):
        """
        Custom validation to check if passwords match.
        """
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")

        if password and password_confirm and password != password_confirm:
            raise ValidationError("Passwords do not match.")

        return cleaned_data
class LoginForm(forms.Form):
    email = forms.EmailField( # Using email as the identifier for Firebase Auth
        label="Email",
        max_length=254,
        widget=forms.EmailInput(attrs={'placeholder': 'Enter your email'})
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={'placeholder': 'Enter your password'})
    )