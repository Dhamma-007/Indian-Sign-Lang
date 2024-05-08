import pickle
import sys
import cv2
import mediapipe as mp
import numpy as np
import io
import streamlit as st
import pyttsx3

def speak(word):
    engine = pyttsx3.init()
    engine.say(word)
    engine.runAndWait()

def main():
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    stframe = st.empty()
    labels_dict = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}
    dt = {"zero": "शून्य", "one": "एक", "two": "दोन", "three": "तीन", "four": "चार"}
    stt=[]
    cap = cv2.VideoCapture(0)
    rows=[st.columns(10) for i in range(20)]
    rc=0
    cc=0
    rno=0
    while True:

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            try:

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                predicted_char=str(predicted_character)
                if len(stt)==0 or  predicted_char != stt[-1]:
                    stt.append(predicted_char)
                    # speak(predicted_char)
                    # print(dt[predicted_char],end=" "
                    # )
                    # st.write(dt[predicted_char])
                    if cc==10:
                        cc=0
                        rc+=1
                    rows[rc][cc].write(dt[predicted_char])
                    cc+=1
                    
                    
                    
                    with open("demofile2.txt", "a", encoding="utf-8") as f:
                        f.write(dt[predicted_char]+" ")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_char, (x1, y1 - 10), font, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            except:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, "two hands detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Frame", frame)

        # Check for 'q' key pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
