from flask import Flask, render_template


import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPool1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, MultiHeadAttention
from keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model


import cv2
import mediapipe as mp


import numpy as np
from threading import Thread
import os
import time
import pyttsx3

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)


@register_keras_serializable(package='CustomModel')
class identity(layers.Layer):
    def __init__(self, filters, kernel_size, name=None, **kwargs):
        super(identity, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv_1 = Conv1D(filters, kernel_size, padding="same", name=f"{name}_conv1" if name else None)
        self.bn_1 = BatchNormalization(name=f"{name}_bn1" if name else None)
        self.act = Activation("relu", name=f"{name}_act" if name else None)
        
        self.conv_2 = Conv1D(filters, kernel_size, padding="same", name=f"{name}_conv2" if name else None)
        self.bn_2 = BatchNormalization(name=f"{name}_bn2" if name else None)
        
        self.add = Add(name=f"{name}_add" if name else None)
        
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.act(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        
        if x.shape[-1] != inputs.shape[-1]:
            inputs = Conv1D(filters=x.shape[-1], kernel_size=1, padding="same")(inputs)
        
        x = self.add([x, inputs])
        x = self.act(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config

@register_keras_serializable(package='CustomModel')
class residual_model(Model):
    def __init__(self, classes, name=None, **kwargs):
        super(residual_model, self).__init__(name=name, **kwargs)
        self.classes = classes
        
        self.conv = Conv1D(64, 7, padding="same")
        self.bn = BatchNormalization()
        self.act = Activation("relu")
        self.pool = MaxPool1D(3)
        
        self.id_1 = identity(64, 3, name="id1")
        self.id_2 = identity(64, 3, name="id2")
        
        self.gape = GlobalAveragePooling1D()
        self.classifier = Dense(classes, activation='softmax')
    
    def call(self, input_val):
        x = self.conv(input_val)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        
        x = self.id_1(x)
        x = self.id_2(x)
        
        x = self.gape(x)
        
        return self.classifier(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "classes": self.classes
        })
        return config


model = load_model("model.keras", custom_objects={"residual_model": residual_model, "identity": identity})

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
               15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: '1', 28: '2'
               , 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}

# Initialize variables
buffer_text = ""
last_predicted_character = ""
gesture_start_time = None
current_character = ""
history_text = ""
current_prediction = None
stability_threshold = 0.1  # Reduced to 0.1 seconds
last_prediction_time = 0
prediction_cooldown = 0.2  # Reduced to 0.2 seconds
prediction_count = 0  # New: Count consecutive frames with the same prediction
min_prediction_count = 3  # New: Require 3 consecutive frames to confirm a prediction

def gen_frames():
    """ Video stream processing """
    global buffer_text, last_prediction_time, gesture_start_time, current_prediction, prediction_count
    cap = None
    for index in range(10):  # Try camera indices 0 through 9
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Using camera at index {index}")
            break
        else:
            cap.release()
    if not cap or not cap.isOpened():
        print("Error: No camera found after trying indices 0-9. Please check camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
    while True:
        try:
            start_time = time.time()
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read frame from camera.")
                break

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                data_aux, x_, y_ = [], [], []

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                data_aux = np.asarray(data_aux, dtype=np.float32).reshape(1, -1, 1)
                current_time = time.time()

                if model is not None and current_time - last_prediction_time >= prediction_cooldown:
                    try:
                        prediction = model.predict(data_aux)
                        predicted_class_index = np.argmax(prediction[0])
                        new_prediction = labels_dict.get(predicted_class_index, "")
                        print(f"Predicted class index: {predicted_class_index}, Character: '{new_prediction}'")

                        if new_prediction:
                            if current_prediction == new_prediction:
                                prediction_count += 1
                                if gesture_start_time is None:
                                    gesture_start_time = current_time
                                    print(f"Gesture started for '{new_prediction}' at {gesture_start_time}")
                                if (current_time - gesture_start_time >= stability_threshold) or (prediction_count >= min_prediction_count):
                                    old_buffer_text = buffer_text
                                    buffer_text += new_prediction
                                    print(f"Stable prediction: '{new_prediction}', old buffer_text: '{old_buffer_text}', new buffer_text: '{buffer_text}', prediction_count: {prediction_count}")
                                    socketio.emit('update_text', {'recognized_text': buffer_text})
                                    print(f"Emitted update_text event with buffer_text: '{buffer_text}'")
                                    gesture_start_time = None
                                    prediction_count = 0  # Reset count after emitting
                            else:
                                current_prediction = new_prediction
                                gesture_start_time = current_time
                                prediction_count = 1  # Reset count for new prediction
                                print(f"New prediction: '{new_prediction}', resetting gesture_start_time to {gesture_start_time}, prediction_count: {prediction_count}")
                        else:
                            current_prediction = None
                            gesture_start_time = None
                            prediction_count = 0
                            print("No valid prediction, resetting current_prediction, gesture_start_time, and prediction_count")

                        last_prediction_time = current_time
                    except Exception as e:
                        print(f"Error during model prediction: {e}")

                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                display_text = current_prediction or ""
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not _:
                print("Error: Failed to encode frame as JPEG.")
                break
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Error in gen_frames: {e}")
            break

    if cap:
        cap.release()
    

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET'])
def get_prediction():
    return jsonify({"recognized_text": buffer_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global buffer_text
    buffer_text = ""
    socketio.emit('update_text', {'recognized_text': buffer_text})
    return jsonify({"status": "cleared"})



@app.route('/test-websocket', methods=['GET'])
def test_websocket():
    global buffer_text
    buffer_text = "TEST"  # Set buffer_text to a test string
    socketio.emit('update_text', {'recognized_text': buffer_text})
    print(f"Manually emitted update_text event with buffer_text: '{buffer_text}'")
    return "WebSocket test emitted"




@app.route('/backspace', methods=['POST'])
def backspace():
    global buffer_text
    print(f"Before backspace: '{buffer_text}'")
    if buffer_text:
        buffer_text = buffer_text[:-1]
    print(f"After backspace: '{buffer_text}'")
    socketio.emit('update_text', {'recognized_text': buffer_text})
    return jsonify({"recognized_text": buffer_text})


@app.route('/save', methods=['POST'])
def save_text():
    data = request.json
    text = data.get('text', '')
    filename = data.get('filename', 'sign_language_output')
    try:
        save_dir = 'static/saved_texts'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{filename}.txt"), 'w', encoding='utf-8') as f:
            f.write(text)
        return jsonify({"status": "saved", "filename": f"{filename}.txt"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

#app
@app.route('/')
def index():
    return render_template('index.html')  # Ensure this returns a valid response

@app.route('/asl_model')
def asl_model():
    return render_template('asl_model.html') 

@socketio.on('connect')
def handle_connect():
    print('Client connected')


if __name__ == '__main__':
    socketio.run(app, debug=True)