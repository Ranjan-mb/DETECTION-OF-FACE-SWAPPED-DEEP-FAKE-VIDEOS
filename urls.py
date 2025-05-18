# detector/urls.py

from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
# Import the new views
from .views import signup_view, login_view, logout_view, about_view, contact_view, legal_view


urlpatterns = [
    path('upload/', views.upload_video, name='upload_video'),
    path('', views.upload_video, name='index'),
    path('signup/', signup_view, name='signup'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    # Add URL patterns for the new pages
    path('about/', about_view, name='about'),
    path('contact/', contact_view, name='contact'),
    path('legal/', legal_view, name='legal'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)