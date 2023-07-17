import cv2

# Load the pre-trained face cascade from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and extract faces from an image
def extract_faces(image_path):
    # Read the image file
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Extract faces and return the results
    extracted_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        extracted_faces.append(face)
    
    return extracted_faces

# Path to the input image
image_path = 'Ritvik.JPG'

# Extract faces from the image
faces = extract_faces(image_path)

# Display or save the extracted faces
for i, face in enumerate(faces):
    cv2.imshow(f"Face {i+1}", face)
    cv2.waitKey(0)

# Close the windows
cv2.destroyAllWindows()
