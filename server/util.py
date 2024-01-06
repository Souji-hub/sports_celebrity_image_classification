import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import os
__class_name_to_number = {}
__class_number_to_name = {}

__model = None


#get b64 encoded value of image
def get_b64_img():
    with open('server/b64.txt') as f:
        return f.read()


#get cv2 image from encoded b64
def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    # Add padding to the base64 string if needed
    padding = len(b64str) % 4
    if padding > 0:
        b64str += '=' * (4 - padding)

    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


#get cropped image
def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('model/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('model/opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        print("Error: Image is not loaded successfully.")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


#load the model
def load_saved_artifacts():
    print("loading saved artifacts...")
    global __class_name_to_number
    global __class_number_to_name

    with open("server/artifacts/class_dict.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('server/artifacts/model.pkl', 'rb') as f:
            __model = joblib.load(f)
    


#classify the image using saved model
def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(int(__model.predict(final)[0])),
            'class_probability': np.around(__model.predict_proba(final)*100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]
    
if __name__ == '__main__':
    
    #testing
    load_saved_artifacts()
    print(classify_image(get_b64_img(), None))  #get image from b64 string
    print(classify_image(None, 'model/test_images/sharapova1.jpg')) #get image from image path
    
    # os.system('cls')
    # data = classify_image(get_b64_img(), None)
    # for entry in data:
    #     print("Class:", entry['class'])
    #     print("Class Probability:", max(entry['class_probability']))
