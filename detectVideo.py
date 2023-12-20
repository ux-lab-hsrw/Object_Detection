import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import cv2
model = load_model('FV.h5')

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']
model = load_model('FV.h5')


# Open a video capture object
video_path = "xxxx"  # Replace with your actual video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame for the model
    frame = cv2.resize(frame, (224, 224))
    frame_for_display = frame.copy()  # Create a copy to display, leaving the original for predictions
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)

    # Make a prediction
    prediction = model.predict(frame)
    class_index = np.argmax(prediction)
    class_label = labels[class_index]

    # Draw text on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (0, 122, 255)  # Orange color
    text = f"{class_label}"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (10, 30)

    cv2.putText(frame_for_display, text, text_position, font, font_scale, font_color, font_thickness)

    # Display the frame with text
    cv2.imshow('Video with Text', frame_for_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()