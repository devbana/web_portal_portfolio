from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


class Cv_chacha:

	def __init__(self):
		self.plant_width = 250
		self.plant_height = 250

	def plant_disease(self, img_path, path_):
		output_dict = {0: 'diseased leaf', 1: 'diseased plant', 2: 'fresh cotton leaf', 3: 'fresh cotton plant'}
		test_image = image.load_img(img_path, target_size=(self.plant_height, self.plant_width))
		prediction_image = np.expand_dims(test_image, axis=0)
		plant_model = load_model(path_ + '/aimodels/plant_disease/plant_vgg.h5')
		result = plant_model.predict(prediction_image)
		ar_result = np.argmax(result)
		# result_list = list(result[0])
		predicted_value = output_dict[int(ar_result)]
		return predicted_value
