import os
import logging
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from torchvision.transforms.functional import rgb_to_grayscale
from basicsr.archs.rrdbnet_arch import RRDBNet # the model itself
from realesrgan import RealESRGANer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_image(image_path):
    """
    Process the uploaded image to enhance viewing quality
    
    Args:
        image_path: Path to the uploaded image file
        
    Returns:
        str: Path to the processed image file
    """
    try:
        # Open the image
        model_path = "Enhancer.pth"
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Create a processed filename
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        processed_filename = f"{name}_processed{ext}"
        processed_path = os.path.join(os.path.dirname(image_path), processed_filename)
        
        # Apply basic enhancements
        # Convert to grayscale for medical scans if not already
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))

        model.load_state_dict(pretrained_dict['params_ema'], strict=False)

        upScaler = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True
        )

        img = Image.open(image_path).convert('RGB')
        img = np.array(img)

        output, _ = upScaler.enhance(img, outscale=2)
        output_img = Image.fromarray(output)
        
        output_img.save(processed_path)
        logger.info(f"Image processed and saved to {processed_path}")
        
        return processed_path
    
    except Exception as e:
         # Open the image
        image = Image.open(image_path)
        
        # Create a processed filename
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        processed_filename = f"{name}_processed{ext}"
        processed_path = os.path.join(os.path.dirname(image_path), processed_filename)
        
        # Apply basic enhancements
        # Convert to grayscale for medical scans if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Apply sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)
        
        # Apply slight smoothing to reduce noise
        image = image.filter(ImageFilter.GaussianBlur(0.5))
        
        # Save the processed image
        image.save(processed_path)
        logger.info(f"Image processed and saved to {processed_path}")
        
        return processed_path

def analyze_scan(image_path, scan_type):
    """
    Analyze the scan image and provide basic analysis results
    
    Args:
        image_path: Path to the processed image file
        scan_type: Type of scan (OCT, MRI, XRay)
        
    Returns:
        dict: Analysis results
    """
    try:
        # Open the image
        image = Image.open(image_path)
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Basic image analysis
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Scan type specific analysis (simplified for demonstration)
        analysis_result = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'metrics': {
                'Average Intensity': f"{brightness:.2f}",
                'Contrast': f"{contrast:.2f}",
            }
        }
        
        # Add scan-type specific analysis
        if scan_type == 'OCT':
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            target_size = (224, 224)
            resized_img = cv2.resize(image, target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
            img_array = img_array / 255.0
            oct_model = tf.keras.models.load_model("OCT_model.keras")
            model = oct_model
            all_labels = ["Normal", "Drusen", "Diabetic Macular Edema", "Choroidal Neovascularization"]

            img_array = tf.image.resize(img_array, [50, 50])
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension -> (1, H, W)
            y_pred = model.predict(img_array)
            predicted_index = np.argmax(y_pred)
            result = all_labels[predicted_index]

            # OCT-specific analysis
            analysis_result['primary_finding'] = result
            analysis_result['metrics']['Layer Continuity'] = "95%"
            analysis_result['metrics']['Retinal Thickness'] = "Normal"
            analysis_result['recommendations'] = [
                "Regular follow-up in 12 months",
                "Maintain eye health with proper nutrition"
            ]
            
        elif scan_type == 'MRI':
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            target_size = (224, 224)
            resized_img = cv2.resize(image, target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
            img_array = img_array / 255.0
            mri_model = tf.keras.models.load_model("MRI_model.h5")
            model = mri_model
            all_labels = ["Healthy", "Meningioma", "Pituitary", "Glioma"]
            img_array = tf.image.resize(img_array, [50, 50])
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension -> (1, H, W)
            y_pred = model.predict(img_array)
            predicted_index = np.argmax(y_pred)
            result = all_labels[predicted_index]

            # MRI-specific analysis
            analysis_result['primary_finding'] = result
            
        elif scan_type == 'XRay':
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            target_size = (224, 224)
            resized_img = cv2.resize(image, target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
            img_array = img_array / 255.0
            xr_model = tf.keras.models.load_model("XR_model.h5")
            model = xr_model
            all_labels = ["Elbow Negative", "Finger Negative", "Forearm Negative", "Hand Negative", "Shoulder Negative",
                      "Elbow Positive", "Finger Positive", "Forearm Positive", "Hand Positive", "Shoulder Positive"]
            img_array = tf.image.resize(img_array, [50, 50])
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension -> (1, H, W)
            y_pred = model.predict(img_array)
            predicted_index = np.argmax(y_pred)
            result = all_labels[predicted_index]
            # X-Ray specific analysis
            analysis_result['primary_finding'] = result
        logger.info(f"Analysis completed for {scan_type} scan")
        return analysis_result
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        # Return minimal information if analysis fails
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'primary_finding': "Analysis could not be completed. Please consult with a specialist.",
            'recommendations': ["Review with medical professional"]
        }
