from emotiondetection.paz.applications import HaarCascadeFrontalFace, MiniXceptionFER
import emotiondetection.paz.processors as pr
import numpy as np


class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()
        self.draw = pr.DrawBoxes2D(self.classify.class_names)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            predictions = self.classify(cropped_image)
            box2D.class_name = predictions['class_name']
            box2D.score = round(np.amax(predictions['scores']),2)
        return boxes2D

    def install(self, image, boxes2D):
        return self.draw(image, boxes2D)
