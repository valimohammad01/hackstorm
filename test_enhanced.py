#!/usr/bin/env python3
"""Test script for enhanced surveillance system"""

def test_imports():
    """Test all required imports"""
    try:
        from detection_engine import DetectionEngine
        print("✅ Detection engine imported successfully")
        
        import streamlit as st
        import cv2
        import numpy as np
        from ultralytics import YOLO
        print("✅ All core dependencies imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_detection_engine():
    """Test detection engine functionality"""
    try:
        from detection_engine import DetectionEngine
        import numpy as np
        
        engine = DetectionEngine()
        print("✅ Detection engine initialized")
        
        # Test with dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        analysis = engine.analyze_frame(dummy_frame)
        
        print(f"✅ Frame analysis completed:")
        print(f"   - Objects: {len(analysis['objects'])}")
        print(f"   - Faces: {len(analysis['faces'])}")
        print(f"   - Alerts: {len(analysis['alerts'])}")
        print(f"   - Processing time: {analysis['processing_time_ms']:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection engine test failed: {e}")
        return False

def main():
    print("🛡️ Enhanced Surveillance AI - Test Suite")
    print("=" * 50)
    
    imports_ok = test_imports()
    detection_ok = test_detection_engine()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Detection Engine: {'✅ PASS' if detection_ok else '❌ FAIL'}")
    
    if imports_ok and detection_ok:
        print("\n🎉 Enhanced surveillance system ready!")
        print("Run: streamlit run core_surveillance.py")
    else:
        print("\n⚠️  Please fix issues before running the system")

if __name__ == "__main__":
    main()
