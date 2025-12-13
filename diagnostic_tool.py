#!/usr/bin/env python3
"""
Face Recognition Diagnostic Tool
Run this to test your face recognition setup and identify issues
"""
import cv2
import numpy as np
from pathlib import Path

# Import your FaceRecognizer (adjust import as needed)
# from your_main_file import FaceRecognizer

def test_face_recognition():
    print("=" * 60)
    print("FACE RECOGNITION DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Check if insightface is available
    print("\n1. Checking insightface availability...")
    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("   ✓ insightface is installed")
        
        try:
            fa = FaceAnalysis(providers=['CPUExecutionProvider'])
            fa.prepare(ctx_id=-1, det_size=(640, 640))
            print("   ✓ FaceAnalysis initialized successfully")
        except Exception as e:
            print(f"   ✗ FaceAnalysis initialization failed: {e}")
            fa = None
    except ImportError:
        print("   ✗ insightface not installed")
        print("   Install with: pip install insightface onnxruntime")
        fa = None
    
    # 2. Check YOLO face model
    print("\n2. Checking YOLO face model...")
    yolo_path = Path("yolov8n-face.pt")
    if yolo_path.exists():
        print(f"   ✓ YOLO face model found: {yolo_path}")
    else:
        print(f"   ✗ YOLO face model not found: {yolo_path}")
        print("   Download from: https://github.com/derronqi/yolov8-face")
    
    # 3. Check faces directory
    print("\n3. Checking faces directory...")
    faces_dir = Path("faces/images")
    if faces_dir.exists():
        images = list(faces_dir.glob("*.jpg")) + list(faces_dir.glob("*.png"))
        print(f"   ✓ Faces directory exists: {faces_dir}")
        print(f"   Found {len(images)} images:")
        for img in images[:10]:  # Show first 10
            print(f"     - {img.name}")
        if len(images) > 10:
            print(f"     ... and {len(images) - 10} more")
    else:
        print(f"   ✗ Faces directory not found: {faces_dir}")
        print("   Create it and add face images")
    
    # 4. Check embeddings database
    print("\n4. Checking embeddings database...")
    db_path = Path("faces/embeddings.json")
    if db_path.exists():
        import json
        try:
            db = json.loads(db_path.read_text())
            print(f"   ✓ Database exists: {db_path}")
            print(f"   Contains {len(db)} people:")
            for name, vecs in db.items():
                print(f"     - {name}: {len(vecs)} templates")
        except Exception as e:
            print(f"   ✗ Database corrupted: {e}")
    else:
        print(f"   ⚠ Database not found: {db_path}")
        print("   It will be created when you register faces")
    
    # 5. Test with a sample image
    print("\n5. Testing face detection and recognition...")
    if faces_dir.exists() and images:
        test_img_path = images[0]
        print(f"   Testing with: {test_img_path.name}")
        
        img = cv2.imread(str(test_img_path))
        if img is None:
            print("   ✗ Could not load image")
        else:
            print(f"   ✓ Image loaded: {img.shape}")
            
            # Test insightface detection
            if fa is not None:
                try:
                    faces = fa.get(img)
                    print(f"   ✓ Insightface detected {len(faces)} face(s)")
                    if faces:
                        for i, face in enumerate(faces):
                            bbox = face.bbox.astype(int)
                            print(f"     Face {i+1}: bbox={bbox}, det_score={face.det_score:.3f}")
                            if hasattr(face, 'embedding'):
                                print(f"              embedding shape={face.embedding.shape}")
                except Exception as e:
                    print(f"   ✗ Insightface detection failed: {e}")
            
            # Test quality checks
            print("\n   Quality checks on image:")
            h, w = img.shape[:2]
            print(f"     Size: {w}x{h}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            print(f"     Blur (Laplacian var): {laplacian_var:.2f} {'✓' if laplacian_var > 50 else '✗ too blurry'}")
            
            # Brightness
            mean_intensity = gray.mean()
            print(f"     Brightness: {mean_intensity:.2f} {'✓' if 20 < mean_intensity < 235 else '✗ bad lighting'}")
            
            # Create test vector
            resized = cv2.resize(gray, (112, 112))
            equalized = cv2.equalizeHist(resized)
            vec = equalized.astype(np.float32).reshape(-1)
            norm = np.linalg.norm(vec)
            print(f"     Vector norm: {norm:.2f} {'✓' if norm > 1e-6 else '✗'}")
    
    # 6. Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if fa is None:
        print("• Install insightface for best accuracy:")
        print("  pip install insightface onnxruntime")
    
    if not yolo_path.exists():
        print("• Download YOLO face detection model:")
        print("  wget https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt")
    
    if not faces_dir.exists() or not images:
        print("• Add face images to faces/images/:")
        print("  - Name format: PersonName.jpg or PersonName_1.jpg")
        print("  - Use clear, well-lit, frontal face photos")
        print("  - Multiple photos per person improves accuracy")
    
    print("\n• To register faces, use the /add_face endpoint or:")
    print("  1. Add images to faces/images/")
    print("  2. Restart the app or use /reload_faces")
    
    print("\n• Enable debug mode in FaceRecognizer for detailed logs")
    print("  self.debug_mode = True")
    
    print("\n" + "=" * 60)

def test_single_image(image_path, recognizer=None):
    """Test recognition on a single image with detailed output"""
    print(f"\nTesting image: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        print("✗ Could not load image")
        return
    
    print(f"✓ Image loaded: {img.shape}")
    
    if recognizer is None:
        # Create temporary recognizer for testing
        from your_main_file import FaceRecognizer  # Adjust import
        recognizer = FaceRecognizer()
    
    # Test detection
    print("\nTesting face detection...")
    faces = recognizer.detect_faces_yolo(img)
    print(f"Detected {len(faces)} faces")
    
    for i, (x1, y1, x2, y2, conf) in enumerate(faces):
        print(f"\nFace {i+1}:")
        print(f"  Bbox: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Confidence: {conf:.3f}")
        print(f"  Size: {x2-x1}x{y2-y1}")
        
        # Extract and test recognition
        crop = img[y1:y2, x1:x2]
        name, score = recognizer.identify(crop)
        print(f"  Identified as: {name}")
        if score is not None:
            print(f"  Score: {score:.4f}")
        
        # Save crop for inspection
        crop_path = f"debug_face_{i+1}.jpg"
        cv2.imwrite(crop_path, crop)
        print(f"  Saved crop to: {crop_path}")
    
    # Visualize
    output = img.copy()
    for x1, y1, x2, y2, conf in faces:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        crop = img[y1:y2, x1:x2]
        name, score = recognizer.identify(crop)
        label = f"{name} ({score:.2f})" if score else name
        cv2.putText(output, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite("debug_output.jpg", output)
    print(f"\nSaved visualization to: debug_output.jpg")

if __name__ == "__main__":
    test_face_recognition()
    
    # Uncomment to test a specific image:
    # test_single_image("path/to/your/test_image.jpg")