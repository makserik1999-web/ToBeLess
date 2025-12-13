from face_recognizer import FaceRecognizer
import cv2, glob, os

fr = FaceRecognizer(debug=True)
files = glob.glob('faces/images/*.*')
if not files:
    print('No images in faces/images')
else:
    f = files[0]
    print('Using file:', f)
    img = cv2.imread(f)
    if img is None:
        print('Failed to load image')
    else:
        boxes = fr.detect_faces(img)
        print('Detected boxes:', boxes)
        if boxes:
            x1,y1,x2,y2,_ = boxes[0]
            crop = img[y1:y2, x1:x2]
            name, score = fr.identify(crop)
            print('identify ->', name, score)
        else:
            # try identify on full image
            name,score = fr.identify(img)
            print('identify(full) ->', name, score)
