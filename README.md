# ASL-Translator-AI

Real-Time Hand Gesture Recogniton and Data Collection

1. Introduction 

This script is designed to capture hand landmarks and images of alphabet hand gestures in real-time using the mediaPipe Hands module. The collected data is organized into directories based on the letters of the alphabet, allowing for the creation of hand gesture datasets for sign langauge.


2. Libraries

OpenCV(cv2): used for capturing video and image processing
mediaPipe(mp): used for hand tracking and landmark detection
numpy(np): employed for numerical operations and data handling 
OS: used for file and directory operation
scikit-learn: Utilized for data preprocessing and splitting 
tensorflow and keras for model trainig 


3. Intitializtaion 

This script initializes the mediaPipe Hands module and a Video capture object.

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 450)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)


4. Gesture Data Collection 

This script collects hand landmarks, draws them on the video frame, and saves the landmarks and cropped hand images for each gesture in separate directories.


while index < len(alphabet):
    # ...
    while True:
        # ...
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # ...
                landmarks_positions = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
                landmarks_file_path = os.path.join(letter_path, letter + str(count) + '.npy')
                np.save(landmarks_file_path, landmarks_positions)
                # ...
                hand_crop = frame[int(bboxC[1]):int(bboxC[1] + bboxC[3]), int(bboxC[0]):int(bboxC[0] + bboxC[2])]
                if hand_crop.size != 0:
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        image_name = letter + str(count) + '.png'
                        image_path = os.path.join(letter_path, image_name)
                        cv2.imwrite(image_path, hand_crop)
                        # ...
                        count += 1
        else:
            print('no hand')
        # ...

5. Data Preprocessing 

Hand gesture images and landmarks are loaded, resized, and normalized for input into the gesture classification model.

for letter in alphabet:
    letter_path = os.path.join(data_path, letter)
    for file_name in os.listdir(letter_path):
        # ...
        hand_landmarks = np.load(landmarks_path, allow_pickle=True)
        image = cv2.imread(image_path)
        # ...
        landmarks_positions = hand_landmarks.flatten()
        images.append(hand_crop_array)
        landmarks_list.append(landmarks_positions)
        labels.append(letter)
        # ...
images = np.array(images)
labels = np.array(labels)
landmarks = np.array(landmarks_list)



6. Gesture Classification Model 

A simple fully connected neural network is built and trained to predict the hand gesture from the landmarks.


model = models.Sequential()
model.add(layers.Flatten(input_shape=(42,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(alphabet), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


7. Model Training and Evaluation 

The model is trained and evaluated on the dataset split into training and testing sets.

history = model.fit(
    x=landmarks_train,
    y=y_train,
    epochs=32,
    batch_size=64,
    validation_data=(landmarks_test, y_test)
)

test_loss, test_acc = model.evaluate(landmarks_test, y_test)
print(f"Test accuracy: {test_acc}")




8. Gesture Recognition

Real-time handlandmarks are predicted using the trained model, and the predicted gesture is displayed on the feed.
  
while index < len(alphabet):
    # ...
    while True:
        # ...
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # ...
                landmarks_positions = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
                landmarks_array = np.array(landmarks_positions).flatten()
                # ...
                predictions = model.predict(landmarks_array.reshape(1, -1))
                predicted_class = np.argmax(predictions)
                cv2.putText(frame, f"Predicted: {alphabet[predicted_class]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # ...

 Conclusion 

This script provides a comprehensive solution for real-time hand gesture recognition making it a valuable tool for application such as sign language translation.


Sources:

1. https://medium.com/mlearning-ai/american-sign-language-alphabet-recognition-ec286915df12
2. https://www.mdpi.com/1424-8220/23/18/7970
3. https://www.kaggle.com/datasets/grassknoted/asl-alphabet
4. https://github.com/computervisioneng/...
   #computervision #signlanguagedetection #objectdetection #scikitlearn #python #opencv #mediapipe #landmarkdetection
5. https://github.com/yuliianikolaenko/asl-alphabet-classification
6. https://github.com/topics/asl-recognizer
7. https://github.com/topics/asl-alphabet-translator
8. https://github.com/11a55an/american-sign-language-detection
9. https://github.com/VedantMistry13/American-Sign-Language-Recognition-using-Deep-Neural-Network
10. https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/main/app.py
11. https://github.com/Kazuhito00/hand-ge...
12. https://www.computervision.zone/cours...
13. https://github.com/nicknochnack/Actio...,  Complete Machine Learning and Data Science Courses
14. chatgpt
15. https://github.com/ivangrov
16. https://www.youtube.com/channel/UCxladMszXan-jfgzyeIMyvw/about
17. https://github.com/nicknochnack/Actio...
 
Application use
	To use the ASL translator clone the git repo at the top of the page. After install the all the previously mentioned libraries. In the file ASL_AI.ipynb run the last block of code. Unblock your webcam if not already done so. Press ‘q’ to quit the application, ‘c’ to capture the image and add it to the sentence, ‘s’ to add a space, and ‘d’ to delete a character.
