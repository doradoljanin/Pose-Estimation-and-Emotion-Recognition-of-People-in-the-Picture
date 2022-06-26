import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from emotiondetection.paz.backend.image import load_image
from emotiondetection.analyzer import EmotionDetector

detect = EmotionDetector()


"""# Returns Boxes of 2D rectangles. Also included is the estimated emotion"""
def emotionFinder(filepath):
    return detect(load_image(filepath))

"""Incorporates the given image with given Boxes of 2D rectangles"""
def incorporate(image, boxes2D):
    # detect = EmotionDetector()
    return detect.install(image, boxes2D)
