import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io
import pytesseract
import hashlib
from scipy.spatial.distance import cosine
import os

# Configure Tesseract path (update based on your installation)
# Windows example:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Mac/Linux: usually in PATH already

class ImagePlagiarismDetector:
    """Advanced Image Plagiarism Detection System"""
    
    @staticmethod
    def load_image(image_file):
        """Load image from uploaded file"""
        try:
            # Read image file
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for OpenCV
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            return image_cv
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    @staticmethod
    def resize_image(image, target_size=(256, 256)):
        """Resize image to standard size for comparison"""
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def calculate_histogram(image):
        """Calculate color histogram for image"""
        # Convert to HSV color space for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Combine histograms
        hist = np.concatenate([hist_h, hist_s, hist_v])
        return hist
    
    @staticmethod
    def histogram_similarity(hist1, hist2):
        """Calculate similarity between two histograms using correlation"""
        # Use correlation method (1 = identical, 0 = different)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        # Convert to percentage
        return max(0, min(100, (similarity + 1) / 2 * 100))
    
    @staticmethod
    def calculate_ssim(image1, image2):
        """Calculate Structural Similarity Index (SSIM)"""
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # Resize to same size for SSIM
        height = min(gray1.shape[0], gray2.shape[0])
        width = min(gray1.shape[1], gray2.shape[1])
        gray1 = cv2.resize(gray1, (width, height))
        gray2 = cv2.resize(gray2, (width, height))
        
        # Compute SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return score * 100
    
    @staticmethod
    def calculate_orb_similarity(image1, image2):
        """Calculate similarity using ORB feature matching"""
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Detect keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)
        
        if des1 is None or des2 is None:
            return 0
        
        # Create BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Calculate similarity based on number of good matches
        num_matches = len(matches)
        max_matches = min(len(kp1), len(kp2))
        
        if max_matches > 0:
            similarity = (num_matches / max_matches) * 100
        else:
            similarity = 0
        
        return min(100, similarity)
    
    @staticmethod
    def calculate_hash_similarity(image):
        """Calculate perceptual hash (pHash) for image"""
        # Resize to 32x32
        resized = cv2.resize(image, (32, 32))
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate DCT
        dct = cv2.dct(np.float32(gray))
        
        # Keep top-left 8x8
        dct_low = dct[:8, :8]
        
        # Calculate median
        median = np.median(dct_low)
        
        # Create hash
        hash_value = (dct_low > median).flatten()
        
        return hash_value
    
    @staticmethod
    def hamming_similarity(hash1, hash2):
        """Calculate similarity between two hashes using Hamming distance"""
        if len(hash1) != len(hash2):
            return 0
        
        hamming_distance = np.sum(hash1 != hash2)
        similarity = (1 - hamming_distance / len(hash1)) * 100
        return similarity
    
    @staticmethod
    def extract_text_from_image(image):
        """Extract text from image using OCR"""
        try:
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to preprocess
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(thresh, lang='eng')
            
            # Clean text
            text = ' '.join(text.split())
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def compare_images(self, img1, img2):
        """Complete image comparison using multiple algorithms"""
        try:
            # Resize images to same dimensions for comparison
            target_size = (256, 256)
            img1_resized = self.resize_image(img1, target_size)
            img2_resized = self.resize_image(img2, target_size)
            
            # 1. Color Histogram Comparison
            hist1 = self.calculate_histogram(img1_resized)
            hist2 = self.calculate_histogram(img2_resized)
            histogram_score = self.histogram_similarity(hist1, hist2)
            
            # 2. SSIM (Structural Similarity)
            ssim_score = self.calculate_ssim(img1_resized, img2_resized)
            
            # 3. ORB Feature Matching
            orb_score = self.calculate_orb_similarity(img1_resized, img2_resized)
            
            # 4. Perceptual Hash
            hash1 = self.calculate_hash_similarity(img1_resized)
            hash2 = self.calculate_hash_similarity(img2_resized)
            hash_score = self.hamming_similarity(hash1, hash2)
            
            # 5. OCR Text Extraction and Comparison
            text1 = self.extract_text_from_image(img1)
            text2 = self.extract_text_from_image(img2)
            text_similarity = self.compare_text_similarity(text1, text2)
            
            # Weighted Final Score
            final_score = (
                histogram_score * 0.20 +
                ssim_score * 0.25 +
                orb_score * 0.20 +
                hash_score * 0.15 +
                text_similarity * 0.20
            )
            
            return {
                "histogram_similarity": round(histogram_score, 2),
                "ssim_similarity": round(ssim_score, 2),
                "orb_similarity": round(orb_score, 2),
                "hash_similarity": round(hash_score, 2),
                "text_similarity": round(text_similarity, 2),
                "final_similarity": round(final_score, 2),
                "extracted_text_img1": text1[:200],  # First 200 chars
                "extracted_text_img2": text2[:200]
            }
        except Exception as e:
            raise Exception(f"Image comparison failed: {str(e)}")
    
    @staticmethod
    def compare_text_similarity(text1, text2):
        """Simple text similarity for OCR results"""
        if not text1 or not text2:
            return 0
        
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = (intersection / union) * 100
        return similarity
    
    def compare_with_repository(self, test_image, repository_images):
        """Compare test image with all images in repository"""
        results = []
        
        for repo_img in repository_images:
            scores = self.compare_images(test_image, repo_img['image'])
            
            results.append({
                "image_id": repo_img['id'],
                "title": repo_img['title'],
                "file_name": repo_img['file_name'],
                "similarity_scores": scores,
                "plagiarism_level": self.get_plagiarism_level(scores['final_similarity'])
            })
        
        # Sort by final similarity (highest first)
        results.sort(key=lambda x: x['similarity_scores']['final_similarity'], reverse=True)
        
        return results
    
    @staticmethod
    def get_plagiarism_level(score):
        if score >= 80:
            return "High Image Plagiarism"
        elif score >= 60:
            return "Moderate Image Plagiarism"
        elif score >= 40:
            return "Low Image Plagiarism"
        elif score >= 20:
            return "Minor Similarity"
        else:
            return "Original Image"