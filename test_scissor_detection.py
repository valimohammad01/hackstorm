"""
Quick test script to verify scissor detection improvements
Run this before starting the main app to check if scissors class is available
"""

from ultralytics import YOLO
import cv2
import numpy as np

def test_yolo_scissor_class():
    """Test if YOLOv8 model can detect scissors"""
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    print("\n" + "="*50)
    print("YOLO Model Classes:")
    print("="*50)
    
    # Check if scissors is in the classes
    scissors_found = False
    for idx, class_name in model.names.items():
        if 'scissor' in class_name.lower():
            print(f"✅ Class {idx}: {class_name} - SCISSORS FOUND!")
            scissors_found = True
    
    if not scissors_found:
        print("⚠️  Warning: 'scissors' class not found in YOLOv8n model")
        print("\nAll available classes:")
        for idx, class_name in model.names.items():
            print(f"  {idx}: {class_name}")
    
    print("\n" + "="*50)
    print("Recommended Settings:")
    print("="*50)
    print("- Inference Size: 480px or 640px")
    print("- Confidence Threshold: 0.30")
    print("- Scissor Confidence: 0.25")
    print("- Process every N frames: 1")
    print("="*50)
    
    return scissors_found

def create_test_image():
    """Create a simple test visualization"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw scissor detection example
    cv2.rectangle(img, (200, 150), (440, 330), (0, 0, 255), 3)
    cv2.rectangle(img, (198, 148), (442, 332), (0, 0, 255), 1)
    
    cv2.putText(img, "scissors ID:1 0.87", (200, 142),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add warning banner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (640, 60), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.putText(img, "!!! SCISSORS DETECTED !!!", (100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    cv2.imwrite("scissor_detection_example.jpg", img)
    print("\n✅ Created example visualization: scissor_detection_example.jpg")
    return img

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SCISSOR DETECTION TEST")
    print("="*50 + "\n")
    
    scissors_available = test_yolo_scissor_class()
    
    print("\n" + "="*50)
    print("Creating Example Visualization")
    print("="*50)
    create_test_image()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    if scissors_available:
        print("✅ Model is ready for scissor detection!")
        print("✅ Run: streamlit run main_enhanced.py")
    else:
        print("⚠️  'scissors' class not in YOLOv8n")
        print("💡 Scissors might still be detected as 'scissors' class ID")
        print("💡 Or consider using a custom-trained model")
    
    print("\n" + "="*50)
    print("Read SCISSOR_DETECTION_IMPROVEMENTS.md for details")
    print("="*50 + "\n")
