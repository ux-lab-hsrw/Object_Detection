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

#img_path="C:\\Users\\philb\\OneDrive\\Desktop\\fruits&veges\\pics\\Mango.jpg" #Replace with your actual imgage file

img_path="sample.png"

model = load_model('FV.h5')
img = load_img(img_path, target_size=(224, 224, 3))
img = img_to_array(img)
img = img / 255
img = np.expand_dims(img, [0])
answer = model.predict(img)
y_class = answer.argmax(axis=-1)
print(y_class)
y = " ".join(str(x) for x in y_class)
y = int(y)
res = labels[y]
print(res)

text = res
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (0, 122, 255)  # orange color


image = cv2.imread(img_path)
# Get the size of the text
text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

# Calculate the position to place the text (top-left corner)
text_position = (10, 30)  # Adjust these coordinates as needed

# Draw the text on the image
cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)

# Display the image with the text
cv2.imshow('Image with Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
