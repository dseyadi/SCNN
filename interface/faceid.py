# import nessesery dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.core.window import Window

import cv2
import tensorflow
from layer import Distance_layer
import os
import numpy as np
from tensorflow.keras.optimizers import RMSprop
import uuid


# build the app
class VerifyApp(App):
    # this function hold the application components
    def build(self): 
        # registration requirements
        self.start_capturing = False
        # path to save the new registration img
        self.test_path = os.path.join('ppl', 'test')
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)

        # register keyboard event
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        # App UX objects
        self.start_label = Label(text="FaceID", size_hint=(1,.1))
        self.web_img = Image(size_hint=(1,.7))
        self.verify_button = Button(text="Verify", on_press=self.verification ,size_hint=(1,.1))
        self.verification_label = Label(text="Uninitiated Verification Process", size_hint=(1,.1))

        # add the UX object to kivy layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.start_label)
        layout.add_widget(self.web_img)
        layout.add_widget(self.verify_button)
        layout.add_widget(self.verification_label)

        #load DL model
        self.model = tensorflow.keras.models.load_model('siamesemodel600.h5', 
                                   custom_objects={'Distance_layer':Distance_layer, 'RMSprop': tensorflow.keras.optimizers.RMSprop, 'BinaryCrossentropy':tensorflow.losses.BinaryCrossentropy}, compile=False)

        # video capture 
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.camRunning, 1.0/33.0) #run it 33 time every second 

        return layout
    

    # this function get the web cam to run continuously
    def camRunning(self, *args):
        ret, frame = self.capture.read() # read the webcam 
        frame = frame[170:170+250, 230:230+250, :]  #cut the frame, discription bellow

        #flip the img and convert it to texture for rendering
        buf = cv2.flip(frame, 0).tostring() #this is a build in function to convert img to buffer
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') #creating textule with specifing the size and color
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte') #convert the buf to apply the opencv img to texture
        self.web_img.texture = img_texture #set the web_img to texture


    #preprocess all images
    def preprocess(self, file_path):
        load_img= tensorflow.io.read_file(file_path) #read the img
        image = tensorflow.io.decode_jpeg(load_img) #load the img
        image = tensorflow.image.resize( image, (105,105)) #resize the img
        image = image / 255.0 #scale the img
        return image
    

    # this function will verify ppl
    def verification(self, *args):
        # set thresholds
        detection_threshold = 0.9
        ver_threshold = 0.99

        # the path to sace the capture web cam image
        save_path = os.path.join('realData', 'inputImage', 'inputImage.jpg')
        ret, frame = self.capture.read() # read the webcam 
        frame = frame[170:170+250, 230:230+250, :]  #cut the frame, discription bellow
        cv2.imwrite(save_path, frame)

        results = []
        for image in os.listdir(os.path.join('realData', 'verifyImage')):
            input_image_path = os.path.join('realData', 'inputImage', 'inputImage.jpg')
            ver_image_path = os.path.join('realData', 'verifyImage', image)
            # Preprocess input and verification images
            input_img = self.preprocess(input_image_path)
            ver_img = self.preprocess(ver_image_path)
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, ver_img], axis=1))) #wrap it into one array because i want one sample 
            results.append(result) #the results will be in one big array
        detectionThreshold = np.sum(np.array(results) > detection_threshold) #how many of the results are a match
        verificationThreshold = detectionThreshold / len(os.listdir(os.path.join('realData', 'verifyImage')))
        # check if the user is verify or not (TRUE / FALSE)
        verified = verificationThreshold > ver_threshold
        # Set verification label 
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'
        # Log out details
        Logger.info(results)
        Logger.info(detectionThreshold)
        Logger.info(verificationThreshold)
        Logger.info(verified)

        return results, verified
    

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        print(f"Key pressed: {text}, Keycode: {keycode}")
        if text == 'r' and keycode[1] == 'r':
            self.start_capturing = not self.start_capturing
            if self.start_capturing:
                Clock.schedule_interval(self.capture_images, 1.0 / 33.0)
        return True

    def capture_images(self, dt):
        if self.start_capturing:
            x = 0
            print(f"Capturing frame {x + 1}")
            while x < 50:  # Keep capturing until 50 frames are captured
                ret, frame = self.capture.read()
                if ret:
                    frame = frame[170:170 + 250, 230:230 + 250, :]
                    imgName = os.path.join(self.test_path, '{}.jpg'.format(uuid.uuid1()))
                    cv2.imwrite(imgName, frame)
                x += 1  # Increment x
            # Stop capturing after 50 frames
            self.start_capturing = False
            Clock.unschedule(self.capture_images)
            self.capture.release()

if __name__ =='__main__':
    VerifyApp().run()