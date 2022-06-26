from emotiondetection.input import emotionFinder, incorporate
from emotiondetection.paz.backend.image import show_image, load_image

def get_emotions(path):
	boxes2D = emotionFinder(path)
	return_list = []
	for j in boxes2D:
		i = j.coordinates
		return_list.append([(i[0],i[1]),(i[2],i[3]),str(j.score) + ' ' + str(j.class_name)]) #xy left lower corner, xy right upper corner, label 
	return(return_list)
# boxes2D = get_emotions('neo.jpg')
# print(boxes2D)
# finalImage = incorporate(image, boxes2D)

# # finalImage = incorporate(image, boxes2D)
# show_image(finalImage)
