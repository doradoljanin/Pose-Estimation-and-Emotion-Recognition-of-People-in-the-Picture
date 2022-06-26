from emotiondetection.input import emotionFinder, incorporate
from emotiondetection.paz.backend.image import show_image, load_image

def get_emotions(path):
	boxes2D = emotionFinder(path)
	image = load_image(path)
	return(boxes2D)

# finalImage = incorporate(image, boxes2D)

# finalImage = incorporate(image, boxes2D)
# show_image(finalImage)
