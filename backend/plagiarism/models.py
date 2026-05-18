from django.db import models
from django.contrib.auth.models import User
from datetime import datetime

class DocumentRepository(models.Model):
    """
    Stores multiple source documents for comparison
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    title = models.CharField(max_length=255)
    content = models.TextField()
    file_name = models.CharField(max_length=255)
    file_size = models.IntegerField(help_text="File size in bytes")
    word_count = models.IntegerField(default=0)
    
    # Preprocessed content for faster comparison
    preprocessed_content = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"

class ComparisonResult(models.Model):
    """
    Stores comparison results between a test document and source documents
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comparisons')
    test_file_name = models.CharField(max_length=255)
    test_content = models.TextField()
    
    # Results
    best_match = models.ForeignKey(DocumentRepository, on_delete=models.SET_NULL, null=True, related_name='matches')
    highest_lcs_score = models.FloatField(default=0)
    highest_tfidf_score = models.FloatField(default=0)
    highest_final_score = models.FloatField(default=0)
    
    # Store all results as JSON
    all_results = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Comparison: {self.test_file_name} - {self.created_at}"
    # Add these new models to your existing models.py

class ImageRepository(models.Model):
    """Store source images for comparison"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='images')
    title = models.CharField(max_length=255)
    image_file = models.ImageField(upload_to='source_images/')
    file_name = models.CharField(max_length=255)
    file_size = models.IntegerField()
    
    # Precomputed features for faster comparison
    color_histogram = models.JSONField(null=True, blank=True)
    perceptual_hash = models.CharField(max_length=255, blank=True)
    extracted_text = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"

class ImageComparisonResult(models.Model):
    """Store image plagiarism check results"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='image_comparisons')
    test_image_name = models.CharField(max_length=255)
    test_image_file = models.ImageField(upload_to='test_images/')
    
    # Results
    best_match = models.ForeignKey(ImageRepository, on_delete=models.SET_NULL, null=True)
    highest_similarity = models.FloatField(default=0)
    all_results = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']