from input import emotionFinder, incorporate
from paz.backend.image import show_image, load_image

image = load_image("neo.png")
boxes2D = emotionFinder(image)

finalImage = incorporate(image, boxes2D)
show_image(finalImage)
quit()
