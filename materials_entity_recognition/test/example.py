import json
import random

from materials_entity_recognition import MatRecognition
from materials_entity_recognition import MatIdentification

# random.seed(datetime.now())
random.seed(7)


if __name__ == "__main__":
	# load data
	with open('../data/test_paras.json', 'r') as fr:
		paras = json.load(fr)

	# load model
	model_new = MatRecognition()
	model_new = MatIdentification()

	# find materials
	for tmp_para in paras:
		# all_materials, precursors, targets, other_materials = model_new.mat_recognize(tmp_para)
		all_materials = model_new.mat_identify(tmp_para)
		print(all_materials['LSTM'])

