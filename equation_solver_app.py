import numpy as np
import keras
import cv2
import hashlib
import hashlib

# Base of a user login system
user_database = {}

def encrypt_password(password):
    salt = "tajna_sól"
    password_with_salt = password + salt
    return hashlib.sha256(password_with_salt.encode()).hexdigest()

def register_user(username, password):
    if username in user_database:
        print("User already exists!")
        return False
    user_database[username] = encrypt_password(password)
    print("User registered successfully!")
    return True

def login_user(username, password):
    if username not in user_database:
        print("User does not exist!")
        return False
    if user_database[username] == encrypt_password(password):
        print("Login successful!")
        return True
    print("Incorrect password!")
    return False

login = input("Provide your login please: ")
password = input("Provide your password please: ")

choice = input("Do you want to register or login? (r/l): ").strip().lower()
try:
    if choice == 'r':
        register_user(login, password)
    if choice == 'l':
        login_user(login, password)
except:
    print("Invalid input!")



# Model and parameters loading
my_model = keras.saving.load_model('equation_solver_v5.keras', custom_objects=None, compile=True, safe_mode=True)
IMG_WIDTH = 45
IMG_HEIGHT = 45
class_names = ['+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
filtered_rectangles = []
resized_images = []
predicted_symbols = []
string_to_int = []
path = input("Please provide the path to your image (e.g. examples/2137plus420.png) : ")



# Equation solving
image = cv2.imread(path, 0)

binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    new_rect = (x, y, x + w, y + h)

    overlap = False
    for rect in filtered_rectangles:
        if (new_rect[0] < rect[2] and new_rect[2] > rect[0] and
                new_rect[1] < rect[3] and new_rect[3] > rect[1]):
            overlap = True
            break

    if not overlap:
        filtered_rectangles.append(new_rect)

contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for rect in filtered_rectangles:
    cv2.rectangle(contour_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
target_size = (45, 45)

for rect in filtered_rectangles:
    x1, y1, x2, y2 = rect
    roi = image[y1:y2, x1:x2]
    resized_roi = cv2.resize(roi, target_size)
    resized_images.append(resized_roi)

for resized_image in resized_images:
    resized_image_normalized = np.expand_dims(resized_image, axis=-1)
    resized_image_normalized = np.expand_dims(resized_image_normalized, axis=0)
    prediction = my_model.predict(resized_image_normalized)
    predicted_class_index = np.argmax(prediction)
    predicted_symbol = class_names[predicted_class_index]
    predicted_symbols.append(predicted_symbol)



# Final equation processing
predicted_string = ''.join(predicted_symbols)
print(f"Predicted string: {predicted_string}")

x = predicted_string.split("+", 1)
for i in x:
    i = int(i)
    string_to_int.append(i)

z = sum(string_to_int)
print(f"Final result of the equation: {z}")