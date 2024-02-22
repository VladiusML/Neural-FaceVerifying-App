from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

from model.SNN import SiameseNN  # Importing Siamese Neural Network model
from scripts.constants import INPUT_PATH, WEIGHTS_PATH  # Importing paths to input images and weights
from scripts.utils import verify_img  # Importing utility function for image verification

import torch
import cv2
import os


class FaceID(App):
    def build(self):
        # Creating widgets
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="", size_hint=(None, None), pos_hint={'center_x': 0.5, 'center_y': 0.7}, opacity=0)

        # Creating layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Loading Siamese Neural Network model
        self.model = SiameseNN()
        self.model.load_state_dict(torch.load(WEIGHTS_PATH))

        # Initializing webcam
        self.capture = cv2.VideoCapture(1)
        # Scheduling update function for webcam feed
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        # Animation for verification label
        anim = Animation(opacity=0, duration=2) + Animation(opacity=1, duration=2)
        anim.repeat = True
        anim.start(self.verification_label)

        return layout
    
    def update(self, *args):
        # Updating webcam feed
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Converting frame to Kivy-compatible texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.7
        verification_threshold = 0.5

        # Capturing frame for verification
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(os.path.join(INPUT_PATH, "input_image.jpg"), frame)

        # Verifying captured image
        results, verified = verify_img(self.model, detection_threshold, verification_threshold)

        # Displaying verification result
        if verified:
            self.verification_label.text = "Verification success"
        else:
            self.verification_label.text = "Verification Error"

        # Animation for verification label
        anim = Animation(opacity=1, duration=0.5) + Animation(opacity=0, duration=0.5)
        anim.start(self.verification_label)

        # Logging results
        Logger.info(results)
        Logger.info(verified)

        return results, verified

if __name__ == '__main__':
    # Running the FaceID application
    FaceID().run()
