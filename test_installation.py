"""
Quick Installation Test Script
Run this to verify all dependencies are installed correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("=" * 60)
    print("Testing Enhanced Surveillance System Installation")
    print("=" * 60)
    print()
    
    tests = []
    
    # Test 1: Streamlit
    print("1. Testing Streamlit...", end=" ")
    try:
        import streamlit as st
        print(f"✅ OK (v{st.__version__})")
        tests.append(True)
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        tests.append(False)
    
    # Test 2: OpenCV
    print("2. Testing OpenCV...", end=" ")
    try:
        import cv2
        print(f"✅ OK (v{cv2.__version__})")
        tests.append(True)
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        tests.append(False)
    
    # Test 3: Ultralytics YOLO
    print("3. Testing Ultralytics YOLO...", end=" ")
    try:
        from ultralytics import YOLO
        print("✅ OK")
        tests.append(True)
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        tests.append(False)
    
    # Test 4: NumPy
    print("4. Testing NumPy...", end=" ")
    try:
        import numpy as np
        print(f"✅ OK (v{np.__version__})")
        tests.append(True)
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        tests.append(False)
    
    # Test 5: Pandas
    print("5. Testing Pandas...", end=" ")
    try:
        import pandas as pd
        print(f"✅ OK (v{pd.__version__})")
        tests.append(True)
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        tests.append(False)
    
    # Test 6: DeepFace
    print("6. Testing DeepFace...", end=" ")
    try:
        from deepface import DeepFace
        print("✅ OK")
        tests.append(True)
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        tests.append(False)
    
    # Test 7: TensorFlow/Keras
    print("7. Testing TensorFlow...", end=" ")
    try:
        import tensorflow as tf
        print(f"✅ OK (v{tf.__version__})")
        tests.append(True)
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        tests.append(False)
    
    # Test 8: Google Generative AI
    print("8. Testing Google Generative AI...", end=" ")
    try:
        import google.generativeai as genai
        print("✅ OK")
        tests.append(True)
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        tests.append(False)
    
    # Test 9: PIL/Pillow
    print("9. Testing Pillow...", end=" ")
    try:
        from PIL import Image
        import PIL
        print(f"✅ OK (v{PIL.__version__})")
        tests.append(True)
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        tests.append(False)
    
    print()
    print("=" * 60)
    
    # Summary
    passed = sum(tests)
    total = len(tests)
    percentage = (passed / total) * 100
    
    print(f"Results: {passed}/{total} tests passed ({percentage:.0f}%)")
    print("=" * 60)
    print()
    
    if passed == total:
        print("🎉 SUCCESS! All dependencies installed correctly!")
        print()
        print("You can now run the enhanced system:")
        print("  streamlit run main_enhanced.py")
        print()
        return True
    elif passed >= 6:
        print("⚠️ PARTIAL SUCCESS - Core features will work")
        print()
        print("Missing optional features:")
        if not tests[5]:  # DeepFace
            print("  - Face expression analysis (DeepFace)")
        if not tests[7]:  # Gemini
            print("  - AI image enhancement (Gemini)")
        print()
        print("You can still run the system with limited features:")
        print("  streamlit run main_enhanced.py")
        print()
        return True
    else:
        print("❌ INSTALLATION INCOMPLETE")
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements_enhanced_v2.txt")
        print()
        return False


def test_files():
    """Check if required files exist"""
    import os
    
    print("=" * 60)
    print("Checking Required Files")
    print("=" * 60)
    print()
    
    required_files = [
        "main_enhanced.py",
        "requirements_enhanced_v2.txt",
        "ENHANCED_SETUP_GUIDE.md",
        "QUICK_START.md"
    ]
    
    optional_files = [
        "yolov8n.pt",
        "demo_weapon.mp4",
        "demo_video.mp4"
    ]
    
    all_good = True
    
    print("Required files:")
    for file in required_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
        if not exists:
            all_good = False
    
    print()
    print("Optional files:")
    for file in optional_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "⚠️"
        print(f"  {status} {file}")
    
    print()
    print("=" * 60)
    
    if all_good:
        print("✅ All required files present")
    else:
        print("❌ Some required files are missing")
    
    print("=" * 60)
    print()
    
    return all_good


def test_camera():
    """Test if camera is available"""
    print("=" * 60)
    print("Testing Camera Access")
    print("=" * 60)
    print()
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("✅ Camera detected and working")
                print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("⚠️ Camera opened but cannot read frames")
                print("   You can still use video files")
        else:
            print("⚠️ No camera detected")
            print("   You can still use video files")
    except Exception as e:
        print(f"❌ Error testing camera: {e}")
    
    print()
    print("=" * 60)
    print()


def main():
    """Run all tests"""
    print()
    print("🔍 Enhanced Surveillance System - Installation Test")
    print()
    
    # Test 1: Package imports
    packages_ok = test_imports()
    
    # Test 2: Required files
    files_ok = test_files()
    
    # Test 3: Camera (optional)
    test_camera()
    
    # Final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print()
    
    if packages_ok and files_ok:
        print("🎉 SYSTEM READY TO USE!")
        print()
        print("Next steps:")
        print("1. (Optional) Get Gemini API key:")
        print("   https://makersuite.google.com/app/apikey")
        print()
        print("2. Run the enhanced system:")
        print("   streamlit run main_enhanced.py")
        print()
        print("3. Or use the batch file (Windows):")
        print("   run_enhanced.bat")
        print()
    elif packages_ok:
        print("⚠️ SYSTEM READY (with warnings)")
        print()
        print("Some files are missing but you can still run:")
        print("  streamlit run main_enhanced.py")
        print()
    else:
        print("❌ INSTALLATION INCOMPLETE")
        print()
        print("Please run:")
        print("  pip install -r requirements_enhanced_v2.txt")
        print()
        print("Then run this test again:")
        print("  python test_installation.py")
        print()
    
    print("=" * 60)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
