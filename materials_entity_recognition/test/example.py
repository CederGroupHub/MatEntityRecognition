import json
import random

from materials_entity_recognition import MatRecognition
from materials_entity_recognition import MatIdentification

# random.seed(datetime.now())
random.seed(7)


if __name__ == "__main__":
	test_paras = []
	with open('../data/test_paras.json', 'r') as fr:
		papers = json.load(fr)
	para_index_dict = {}
	for tmp_paper in papers:
		for tmp_para in tmp_paper['paragraphs']:
			if tmp_para['recipeOrNot'] == "Y":
				test_paras.append({'text': tmp_para['text'], 'from_paper': tmp_paper['doi']})
				if tmp_paper['doi'] not in para_index_dict:
					para_index_dict[tmp_paper['doi']] = []
				para_index_dict[tmp_paper['doi']].append(len(test_paras)-1)
	print('len(test_paras)', len(test_paras))
	# load model
	model_new = MatRecognition()
	model_new = MatIdentification()
	
	input_paras = test_paras.copy()
	random.shuffle(input_paras)
	for tmp_index, tmp_para in enumerate(input_paras[0:]):
		if(tmp_index%100) == 0:
			print(tmp_index)

		tmp_para_text = tmp_para['text']


		# all_materials, precursors, targets, other_materials = model_new.mat_recognize(tmp_para_text)
		all_materials = model_new.mat_identify(tmp_para_text)
		print(all_materials['LSTM'])

