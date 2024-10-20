import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./my_yoga_model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
cap.set(3,2000)
cap.set(4,2000)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {0: 'Tree', 1: 'Goddess', 2: 'Mountain',3: 'Chair Pose',4: 'Cobra Pose',5: 'Diamond Pose'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,  
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS, 
            mp_drawing_styles.get_default_pose_landmarks_style())

        for i in range(len(results.pose_landmarks.landmark)):
            x = results.pose_landmarks.landmark[i].x
            y = results.pose_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(results.pose_landmarks.landmark)):
            x = results.pose_landmarks.landmark[i].x
            y = results.pose_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_pose = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_pose, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
