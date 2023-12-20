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

# Open a webcam capture object (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop through each frame from the webcam
while True:
    ret, frame = cap.read()

    # Preprocess the frame for the model
    frame_for_display = frame.copy()  # Create a copy to display, leaving the original for predictions
    frame = cv2.resize(frame, (224, 224))
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
    text = f"Prediction: {class_label}"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (10, 30)

    cv2.putText(frame_for_display, text, text_position, font, font_scale, font_color, font_thickness)

    # Display the frame with text
    cv2.imshow('Webcam with Text', frame_for_display)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam capture object
cap.release()
cv2.destroyAllWindows()
