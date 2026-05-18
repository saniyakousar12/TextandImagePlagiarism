from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from .models import DocumentRepository, ComparisonResult

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2']

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data['email']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user

class DocumentRepositorySerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentRepository
        fields = ['id', 'title', 'file_name', 'file_size', 'word_count', 'created_at']
        read_only_fields = ['id', 'created_at', 'word_count']

class DocumentDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentRepository
        fields = ['id', 'title', 'content', 'file_name', 'file_size', 'word_count', 'created_at']

class ComparisonResultSerializer(serializers.ModelSerializer):
    best_match_title = serializers.CharField(source='best_match.title', read_only=True)
    
    class Meta:
        model = ComparisonResult
        fields = ['id', 'test_file_name', 'highest_lcs_score', 'highest_tfidf_score', 
                  'highest_final_score', 'best_match_title', 'all_results', 'created_at']