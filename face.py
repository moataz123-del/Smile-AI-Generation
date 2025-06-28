import message as Message
import helper as Helper
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

from skimage.io import imread, imshow
from skimage.filters import prewitt_h, prewitt_v

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables initialization
mouthPoints = None
mouth_left_x = None
mouth_right_x = None
mouth_center_x = None
mouth_center_y = None
mouth_center_y2 = None
mouth_center_y3 = None
eyes_center_x = None
eyes_center_y = None
img = None
results = ""
incisor_width = None
final_midline = None
incisors_lower_edge = None
incisor = None
mouthImage = None
gummy_smile = False
x = None
y = None
current_template_scale = 1.0
current_template_shape = None

# File paths
mouthImagePath = "cached/mouth.png"
finalImagePath = "cached/final.png"
templatePath = {
    "Square": "cached/square.png",
    "Rectangle": "cached/rectangle.png",
    "Triangle": "cached/triangle.png",
    "Oval": "cached/oval.png",
}
imagePath = "cached/image.png"
templateImagePath = "cached/template.png"


def mouthDetection(file_path):
    global mouthPoints, mouth_left_x, mouth_right_x, mouth_center_x, mouth_center_y
    global mouth_center_y2, mouth_center_y3, eyes_center_x, eyes_center_y, img

    img = cv2.imread(file_path)
    
    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")
        
        # Get the first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # MediaPipe face mesh has 468 landmarks
        # Mouth landmarks (approximate mapping from dlib's 68-point model)
        # Inner mouth points (60-67 in dlib) correspond to MediaPipe landmarks around 61, 84, 17, 314, 405, 320, 307, 375
        mouth_landmarks = [61, 84, 17, 314, 405, 320, 307, 375]
        
        # Extract mouth points
        xmouthpoints = []
        ymouthpoints = []
        for landmark_id in mouth_landmarks:
            landmark = face_landmarks.landmark[landmark_id]
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            xmouthpoints.append(x)
            ymouthpoints.append(y)
        
        # Get eye center (between left and right eye)
        left_eye = face_landmarks.landmark[33]  # Left eye center
        right_eye = face_landmarks.landmark[263]  # Right eye center
        eyes_center_x = int((left_eye.x + right_eye.x) * img.shape[1] / 2)
        eyes_center_y = int((left_eye.y + right_eye.y) * img.shape[0] / 2)
        
        # Get mouth center and boundaries
        # Mouth center (between upper and lower lip)
        upper_lip = face_landmarks.landmark[13]  # Upper lip center
        lower_lip = face_landmarks.landmark[14]  # Lower lip center
        mouth_center_x = int((upper_lip.x + lower_lip.x) * img.shape[1] / 2)
        mouth_center_y = int((upper_lip.y + lower_lip.y) * img.shape[0] / 2)
        mouth_center_y2 = int(upper_lip.y * img.shape[0])
        mouth_center_y3 = int(lower_lip.y * img.shape[0])
        
        # Mouth left and right boundaries
        mouth_left = face_landmarks.landmark[61]  # Left corner of mouth
        mouth_right = face_landmarks.landmark[291]  # Right corner of mouth
        mouth_left_x = int(mouth_left.x * img.shape[1])
        mouth_right_x = int(mouth_right.x * img.shape[1])

    pts = []
    for i in range(0, 8):
        pts.append([xmouthpoints[i], ymouthpoints[i]])

    mouthPoints = np.array(pts)
    mouthCrop(img, mouthPoints)
    cv2.imwrite(imagePath, img)


def mouthCrop(img, mouthPoints):
    global x, y
    rect = cv2.boundingRect(mouthPoints)
    x, y, w, h = rect
    croped = img[y : y + h, x : x + w].copy()
    mouthPoints = mouthPoints - mouthPoints.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [mouthPoints], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    cv2.imwrite(mouthImagePath, dst)


def checkMidline():
    global results, incisor_width, final_midline, incisors_lower_edge, incisor

    ratio = int(mouth_right_x - int(mouth_left_x))
    ratio = int(ratio / 6)

    midline = []
    final_midlines = []
    shiftFlag = True
    img = cv2.imread(imagePath)

    for i in range(-1 * ratio, ratio):
        bgr = np.array(img[mouth_center_y][mouth_center_x + i])
        midline.append([bgr[0], mouth_center_x + i])

    midline.sort()

    for idx, x in enumerate(midline):
        for elem in midline[idx + 1 :]:
            if (elem[1] < x[1] + 10) and (elem[1] > x[1] - 10):
                midline.remove(elem)
    for i in range(0, 3):
        final_midlines.append(midline[i])
    final_midlines.sort(key=lambda x: x[1])

    distances = []
    for i in range(1, 3):
        distances.append(abs(final_midlines[i][1] - final_midlines[i - 1][1]))

    incisor_width = 0
    if abs(distances[0] - distances[1]) <= 5:
        incisor_width = distances[0]
        final_midline = final_midlines[1][1]
    elif distances[0] > distances[1]:
        final_midline = final_midlines[0][1]
        incisor_width = distances[0]
    else:
        final_midline = final_midlines[2][1]
        incisor_width = distances[1]

    if abs(final_midline - eyes_center_x) <= 8:
        shiftFlag = False
    else:
        shiftFlag = True

    if shiftFlag:
        results += "A midline shift found\n"
    else:
        results += "Facial and Dental midline are almost identical. No shift found\n"

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    pixel = edges[mouth_center_y][final_midline + int(incisor_width / 2)]

    up = 0
    while int(pixel) == 0:
        up += 1
        pixel = edges[mouth_center_y + up][final_midline + int(incisor_width / 2)]

    image = cv2.line(
        img,
        (final_midline + int(incisor_width / 2), mouth_center_y + up),
        (final_midline + int(incisor_width / 2) + 200, mouth_center_y + up),
        color=(0, 20, 0),
        thickness=2,
    )
    incisors_lower_edge = mouth_center_y + up - int(1.25 * incisor_width)
    incisor = mouth_center_y + up


def applyTemplateToResult(shape):
    global current_template_shape, current_template_scale
    
    if mouth_right_x is None or mouth_left_x is None or incisor_width is None:
        raise Exception("Please run mouth detection first")
    
    # Load the original image and template
    original_img = Image.open(imagePath)
    template_img = Image.open(templatePath[shape])
    
    # Calculate template size
    mouth_width = mouth_right_x - mouth_left_x
    template_width = int(mouth_width * 0.8 * current_template_scale)
    template_height = int(1.25 * incisor_width * current_template_scale)
    
    # Resize template
    template_img = template_img.resize((template_width, template_height))
    
    # Calculate position to overlay template
    template_x = final_midline - int(template_width / 2)
    template_y = incisors_lower_edge
    
    # Convert to RGBA for transparency
    if template_img.mode != 'RGBA':
        template_img = template_img.convert('RGBA')
    
    # Create a copy of the original image
    result_img = original_img.copy()
    if result_img.mode != 'RGBA':
        result_img = result_img.convert('RGBA')
    
    # Paste template onto result image
    result_img.paste(template_img, (template_x, template_y), template_img)
    
    # Save the result
    result_img.save(finalImagePath)
    current_template_shape = shape


def scaleCurrentTemplate(scale):
    global current_template_scale
    
    if current_template_shape is None:
        raise Exception("No template applied yet")
    
    current_template_scale *= scale
    applyTemplateToResult(current_template_shape)


def checkGummySmile():
    global results, mouthImage, gummy_smile

    mouthImage = cv2.imread(mouthImagePath)
    redCount = 0
    blackCount = 0
    rows, cols, _ = mouthImage.shape
    for i in range(rows):
        for j in range(cols):
            pixel = mouthImage[i, j]
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                blackCount += 1
            else:
                if pixel[0] < 150 and pixel[1] < 150 and pixel[2] > 200:
                    redCount += 1

    ratio = redCount / ((rows * cols) - blackCount)
    if ratio > 0.07:
        results += "There is a gummy smile\n"
        gummy_smile = True
    else:
        results += "There is no gummy smile\n"
        gummy_smile = False


def checkDiastema():
    global results, img, incisor, gummy_smile

    img5 = cv2.imread(Helper.filePath, 0)
    img6 = img5
    gap = 0
    row = 0
    for i in range(0, 10):
        if gummy_smile:
            for j in range(-5, 5):
                pixel_color = np.array(
                    img6[mouth_center_y2 + 16 + i][mouth_center_x + j]
                )

                if mouth_center_y2 + 16 + i > incisor:
                    break
                if pixel_color < 120:
                    gap += 1
            if gap > 2:
                row += 1
                gap = 0

        else:
            for j in range(-5, 5):
                pixel_color = np.array(
                    img6[
                        mouth_center_y2
                        + int((mouth_center_y3 - mouth_center_y2) / 4)
                        + i
                    ][mouth_center_x + j]
                )
                if (
                    mouth_center_y2 + int((mouth_center_y3 - mouth_center_y2) / 4) + i
                    > incisor
                ):
                    break
                if pixel_color < 120:
                    gap += 1
            if gap > 2:
                row += 1
                gap = 0

    if row > 2:
        results += "There is a diastema"
    else:
        results += "There is no diastema"


def checkAll():
    global results

    results = ""
    checkGummySmile()
    checkMidline()
    checkDiastema()
    
    # Copy the original image to final image path for initial display
    import shutil
    shutil.copy(imagePath, finalImagePath)


def showResults():
    global results
    Message.info(results)
