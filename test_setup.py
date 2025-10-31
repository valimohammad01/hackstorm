#!/usr/bin/env python3
"""Test script to verify optimized surveillance setup"""

def test_dependencies():
    """Test all required dependencies"""
    print("🔍 Testing Dependencies...")
    
    try:
        import streamlit
        print(f"✅ Streamlit: {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        print(f"✅ System Monitor: OK")
        print(f"💻 RAM: {memory_gb:.1f} GB")
        print(f"💻 CPU Cores: {cpu_count}")
    except ImportError as e:
        print(f"❌ psutil: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print(f"✅ Ultralytics YOLO: OK")
    except ImportError as e:
        print(f"❌ Ultralytics: {e}")
        return False
    
    return True

def test_model_loading():
    """Test YOLO model loading"""
    print("\n🤖 Testing Model Loading...")
    
    try:
        from ultralytics import YOLO
        import os
        
        model_path = "yolov8n.pt"
        if os.path.exists(model_path):
            print(f"✅ Model file found: {model_path}")
            
            # Test loading
            model = YOLO(model_path)
            print("✅ Model loaded successfully")
            
            # Test inference on dummy data
            import torch
            dummy_input = torch.randn(1, 3, 640, 640)
            
            with torch.no_grad():
                results = model(dummy_input, verbose=False)
            
            print("✅ Model inference test passed")
            return True
            
        else:
            print(f"❌ Model file not found: {model_path}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("\n📹 Testing Camera Access...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera working - Frame size: {frame.shape}")
                cap.release()
                return True
            else:
                print("❌ Camera not returning frames")
                cap.release()
                return False
        else:
            print("❌ Cannot open camera")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def performance_recommendations():
    """Provide performance recommendations"""
    print("\n🚀 Performance Recommendations...")
    
    try:
        import psutil
        import torch
        
        memory_gb = psutil.virtual_memory().total / (1024**3)
        has_gpu = torch.cuda.is_available()
        cpu_count = psutil.cpu_count()
        
        if memory_gb >= 16 and has_gpu:
            tier = "HIGH"
            settings = "Inference: 640px, Frame Skip: 2, Target FPS: 20"
        elif memory_gb >= 8:
            tier = "MEDIUM" 
            settings = "Inference: 480px, Frame Skip: 3, Target FPS: 15"
        else:
            tier = "LOW"
            settings = "Inference: 320px, Frame Skip: 4, Target FPS: 10"
        
        print(f"🎯 Recommended Performance Tier: {tier}")
        print(f"⚙️  Optimal Settings: {settings}")
        print(f"🔧 Device: {'GPU' if has_gpu else 'CPU'}")
        
        if memory_gb < 8:
            print("⚠️  Warning: Low RAM detected. Consider closing other applications.")
        
        if not has_gpu:
            print("💡 Tip: GPU not available. CPU mode will be used (slower but functional).")
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")

def main():
    """Main test function"""
    print("🛡️ Optimized AI Surveillance - Setup Test")
    print("=" * 50)
    
    deps_ok = test_dependencies()
    model_ok = test_model_loading()
    camera_ok = test_camera()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Camera Access: {'✅ PASS' if camera_ok else '❌ FAIL'}")
    
    if deps_ok and model_ok:
        print("\n🎉 System ready for optimized surveillance!")
        print("Run: streamlit run optimized_surveillance.py")
    else:
        print("\n⚠️  Please fix the issues above before running the surveillance system.")
    
    performance_recommendations()

if __name__ == "__main__":
    main()
