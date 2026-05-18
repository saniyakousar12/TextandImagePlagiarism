from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework_simplejwt.tokens import RefreshToken
from .text_detection import compare_with_repository, preprocess, calculate_similarity
from .serializers import (
    RegisterSerializer, UserSerializer, 
    DocumentRepositorySerializer, DocumentDetailSerializer,
    ComparisonResultSerializer
)
from .models import DocumentRepository, ComparisonResult
from .image_detection import ImagePlagiarismDetector
from .models import ImageRepository, ImageComparisonResult
import numpy as np

def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }

# ========== AUTHENTICATION VIEWS ==========

@api_view(['GET'])
def test_api(request):
    return Response({
        "message": "Backend Connected Successfully"
    })

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        return Response({
            "message": "User created successfully",
            "user": UserSerializer(user).data
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    email = request.data.get('email')
    password = request.data.get('password')
    
    if not email or not password:
        return Response({
            "message": "Please provide both email and password"
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        user = User.objects.get(email=email)
    except User.DoesNotExist:
        return Response({
            "message": "Invalid credentials"
        }, status=status.HTTP_401_UNAUTHORIZED)
    
    user = authenticate(username=user.username, password=password)
    
    if user:
        tokens = get_tokens_for_user(user)
        return Response({
            "message": "Login successful",
            "access": tokens['access'],
            "refresh": tokens['refresh'],
            "user": UserSerializer(user).data
        })
    
    return Response({
        "message": "Invalid credentials"
    }, status=status.HTTP_401_UNAUTHORIZED)

# ========== DOCUMENT REPOSITORY VIEWS ==========

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_source_document(request):
    """Upload a document to the source repository"""
    try:
        file = request.FILES.get('file')
        title = request.data.get('title', file.name)
        
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Read file content
        content = file.read().decode('utf-8')
        
        # Preprocess for faster future comparisons
        preprocessed_content = preprocess(content)
        
        document = DocumentRepository.objects.create(
            user=request.user,
            title=title,
            content=content,
            file_name=file.name,
            file_size=file.size,
            word_count=len(content.split()),
            preprocessed_content=preprocessed_content
        )
        
        serializer = DocumentRepositorySerializer(document)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_source_documents(request):
    """List all source documents for the user"""
    documents = DocumentRepository.objects.filter(user=request.user)
    serializer = DocumentRepositorySerializer(documents, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_document_detail(request, document_id):
    """Get detailed content of a specific document"""
    try:
        document = DocumentRepository.objects.get(id=document_id, user=request.user)
        serializer = DocumentDetailSerializer(document)
        return Response(serializer.data)
    except DocumentRepository.DoesNotExist:
        return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_source_document(request, document_id):
    """Delete a source document from repository"""
    try:
        document = DocumentRepository.objects.get(id=document_id, user=request.user)
        document.delete()
        return Response({"message": "Document deleted successfully"})
    except DocumentRepository.DoesNotExist:
        return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)

# ========== PLAGIARISM CHECKING VIEWS ==========

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def check_against_repository(request):
    """Check a test document against all source documents"""
    try:
        test_file = request.FILES.get('test_file')
        
        if not test_file:
            return Response({"error": "No test file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Read test document
        test_content = test_file.read().decode('utf-8')
        
        # Get all source documents for this user
        source_docs = DocumentRepository.objects.filter(user=request.user)
        
        if not source_docs.exists():
            return Response({
                "error": "No source documents in repository. Please upload source documents first."
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Prepare repository documents list
        repository = [
            {
                'id': doc.id,
                'title': doc.title,
                'content': doc.content,
                'file_name': doc.file_name
            }
            for doc in source_docs
        ]
        
        # Compare with all documents
        results = compare_with_repository(test_content, repository)
        
        # Save comparison result
        best_match = results[0] if results else None
        comparison = ComparisonResult.objects.create(
            user=request.user,
            test_file_name=test_file.name,
            test_content=test_content,
            highest_lcs_score=best_match['lcs_score'] if best_match else 0,
            highest_tfidf_score=best_match['tfidf_score'] if best_match else 0,
            highest_final_score=best_match['final_score'] if best_match else 0,
            best_match_id=best_match['document_id'] if best_match else None,
            all_results=results
        )
        
        return Response({
            "comparison_id": comparison.id,
            "total_documents_compared": len(results),
            "best_match": results[0] if results else None,
            "all_results": results[:10],  # Top 10 results
            "summary": {
                "highest_match": results[0]['final_score'] if results else 0,
                "plagiarism_level": results[0]['plagiarism_level'] if results else "No matches"
            }
        })
    
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_comparison_history(request):
    """Get user's comparison history"""
    comparisons = ComparisonResult.objects.filter(user=request.user)
    serializer = ComparisonResultSerializer(comparisons, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_comparison_detail(request, comparison_id):
    """Get detailed results of a specific comparison"""
    try:
        comparison = ComparisonResult.objects.get(id=comparison_id, user=request.user)
        serializer = ComparisonResultSerializer(comparison)
        return Response(serializer.data)
    except ComparisonResult.DoesNotExist:
        return Response({"error": "Comparison not found"}, status=status.HTTP_404_NOT_FOUND)

# ========== IMAGE PLAGIARISM VIEWS ==========

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_source_image(request):
    """Upload source image to repository"""
    try:
        image_file = request.FILES.get('image')
        title = request.data.get('title', image_file.name)
        
        if not image_file:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
        if image_file.content_type not in allowed_types:
            return Response({"error": "Only JPEG, PNG, and BMP images are allowed"}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        # Load and process image
        detector = ImagePlagiarismDetector()
        image = detector.load_image(image_file)
        image_file.seek(0)  # Reset file pointer
        
        # Extract features for faster future comparisons
        image_resized = detector.resize_image(image)
        hist = detector.calculate_histogram(image_resized)
        hash_val = detector.calculate_hash_similarity(image_resized)
        extracted_text = detector.extract_text_from_image(image)
        
        # Save to database
        image_repo = ImageRepository.objects.create(
            user=request.user,
            title=title,
            image_file=image_file,
            file_name=image_file.name,
            file_size=image_file.size,
            color_histogram=hist.tolist() if isinstance(hist, np.ndarray) else hist,
            perceptual_hash=hash_val.tobytes().hex() if isinstance(hash_val, np.ndarray) else str(hash_val),
            extracted_text=extracted_text
        )
        
        return Response({
            "message": "Image uploaded successfully",
            "id": image_repo.id,
            "title": image_repo.title,
            "file_name": image_repo.file_name
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_source_images(request):
    """List all source images"""
    images = ImageRepository.objects.filter(user=request.user)
    data = [{
        "id": img.id,
        "title": img.title,
        "file_name": img.file_name,
        "file_size": img.file_size,
        "created_at": img.created_at,
        "image_url": img.image_file.url if img.image_file else None
    } for img in images]
    return Response(data)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_source_image(request, image_id):
    """Delete source image from repository"""
    try:
        image = ImageRepository.objects.get(id=image_id, user=request.user)
        # Delete the actual file
        if image.image_file:
            image.image_file.delete()
        image.delete()
        return Response({"message": "Image deleted successfully"})
    except ImageRepository.DoesNotExist:
        return Response({"error": "Image not found"}, status=status.HTTP_404_NOT_FOUND)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def check_image_plagiarism(request):
    """Check test image against all source images"""
    try:
        test_image = request.FILES.get('test_image')
        
        if not test_image:
            return Response({"error": "No test image provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get all source images
        source_images = ImageRepository.objects.filter(user=request.user)
        
        if not source_images.exists():
            return Response({
                "error": "No source images in repository. Please upload source images first."
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Load test image
        detector = ImagePlagiarismDetector()
        test_img = detector.load_image(test_image)
        
        # Prepare repository
        repository = []
        for img in source_images:
            # Load image from file
            img_file = img.image_file
            img_file.seek(0)
            repo_img = detector.load_image(img_file)
            repository.append({
                'id': img.id,
                'title': img.title,
                'image': repo_img,
                'file_name': img.file_name
            })
        
        # Compare with all images
        results = detector.compare_with_repository(test_img, repository)
        
        # Save comparison result
        best_match = results[0] if results else None
        comparison = ImageComparisonResult.objects.create(
            user=request.user,
            test_image_name=test_image.name,
            test_image_file=test_image,
            highest_similarity=best_match['similarity_scores']['final_similarity'] if best_match else 0,
            best_match_id=best_match['image_id'] if best_match else None,
            all_results=results
        )
        
        # Reset file pointer
        test_image.seek(0)
        
        return Response({
            "comparison_id": comparison.id,
            "total_images_compared": len(results),
            "best_match": results[0] if results else None,
            "all_results": results[:10],
            "summary": {
                "highest_similarity": results[0]['similarity_scores']['final_similarity'] if results else 0,
                "plagiarism_level": results[0]['plagiarism_level'] if results else "No matches"
            }
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_image_comparison_history(request):
    """Get image comparison history"""
    comparisons = ImageComparisonResult.objects.filter(user=request.user)
    data = [{
        "id": comp.id,
        "test_image_name": comp.test_image_name,
        "highest_similarity": comp.highest_similarity,
        "created_at": comp.created_at,
        "best_match_title": comp.best_match.title if comp.best_match else None
    } for comp in comparisons]
    return Response(data)