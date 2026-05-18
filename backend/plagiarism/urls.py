from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Test
    path('test/', views.test_api, name='test_api'),
    
    # Authentication
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    
    # Document Repository
    path('documents/upload/', views.upload_source_document, name='upload_source'),
    path('documents/list/', views.list_source_documents, name='list_documents'),
    path('documents/<int:document_id>/delete/', views.delete_source_document, name='delete_document'),
    path('check/', views.check_against_repository, name='check_plagiarism'),
    path('comparisons/', views.get_comparison_history, name='comparison_history'),
    path('comparisons/<int:comparison_id>/', views.get_comparison_detail, name='comparison_detail'),
    
    # Image Plagiarism
    path('images/upload/', views.upload_source_image, name='upload_source_image'),
    path('images/list/', views.list_source_images, name='list_source_images'),
    path('images/<int:image_id>/delete/', views.delete_source_image, name='delete_source_image'),
    path('check-image/', views.check_image_plagiarism, name='check_image_plagiarism'),
    path('image-comparisons/', views.get_image_comparison_history, name='image_comparison_history'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)