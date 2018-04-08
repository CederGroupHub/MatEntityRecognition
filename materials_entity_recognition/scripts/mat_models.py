import numpy as np
import chemdataextractor as CDE
import os

from .model import Model
from .loader import prepare_sentence
from .utils import create_input, iobes_iob, zero_digits


class MatIdentification():
	"""
	Use LSTM for materials identification
	"""
	def __init__(self, model_path = None):
		if model_path == None:
			file_path = os.path.dirname(__file__)
			self.model = Model(model_path = os.path.join(file_path, '..', 'models/matIdentification'))		
		else:
			self.model = Model(model_path = model_path)
		parameters = self.model.parameters
		word_to_id, char_to_id, tag_to_id = [
		    {v: k for k, v in list(x.items())}
		    for x in [self.model.id_to_word, self.model.id_to_char, self.model.id_to_tag]
		]
		self.parameters = parameters
		self.word_to_id = word_to_id
		self.char_to_id = char_to_id
		self.tag_to_id = tag_to_id
		_, f_eval = self.model.build(training = False, **self.parameters)
		self.f_eval = f_eval
		self.model.reload()

	def mat_identify_sent(self, input_sent):
		"""
		Identify materials in a sentence, which is a list of tokens.
		Input: list of tokens representing a sentence 
		Output: list of materials from LSTM
		"""
		# goal
		materials = []
		# Prepare input
		words = [tmp_token['text'] for tmp_token in input_sent]
		sentence = prepare_sentence(words, self.word_to_id, self.char_to_id, \
									lower=self.parameters['lower'])
		input = create_input(sentence, self.parameters, False)
		# Prediction
		if self.parameters['crf']:
		    y_preds = np.array(self.f_eval(*input))[1:-1]
		else:
		    y_preds = self.f_eval(*input).argmax(axis = 1)
		y_preds = [self.model.id_to_tag[y_pred] for y_pred in y_preds]
		y_preds = iobes_iob(y_preds)
		mat_begin = False
		for tmp_index, y_pred in enumerate(y_preds):
			if y_pred == 'B-Mat':
				materials.append(input_sent[tmp_index])
				mat_begin = True
			elif y_pred == 'I-Mat' and mat_begin == True:
				materials[-1]['end'] = input_sent[tmp_index]['end']
				materials[-1]['text'] += ' ' + input_sent[tmp_index]['text']
			else:
				mat_begin = False
		return materials

	def mat_identify(self, input_para):
		"""
		Identify materials in a paragraph, which is plain text.
		Input: str representing a paragraph
		Output: dict containing materials from CDE (dict['CDE']) and materials from LSTM (dict['LSTM'])
		"""
		# goal
		materials = {}
		# CDE tokenization
		CDE_para = CDE.doc.Paragraph(input_para)
		materials['CDE'] = [{'text': tmp_cem.text, 'start': tmp_cem.start, 'end': tmp_cem.end} \
							for tmp_cem in CDE_para.cems]
		materials['LSTM'] = []
		for tmp_sent in CDE_para:
			# prepare input sentences for LSTM
			input_sent = [{'text': tmp_token.text, 'start': tmp_token.start, 'end': tmp_token.end, 'sentence': tmp_sent.text} \
							for tmp_token in tmp_sent.tokens]
			input_sent = list(filter(lambda tmp_token: tmp_token['text'].strip() != '', input_sent))
			# use LSTM to identify materials
			materials['LSTM'].extend(self.mat_identify_sent(input_sent))
		# reformated as the exact words in the original paragraph
		for tmp_mat in materials['LSTM']:
			tmp_mat['text'] = input_para[tmp_mat['start']: tmp_mat['end']]

		return materials

class MatRecognition():
	"""
	Use LSTM for materials recognition
	"""
	def __init__(self, model_path = None, mat_identify_model_path = None):
		if model_path == None:
			file_path = os.path.dirname(__file__)
			self.model = Model(model_path = os.path.join(file_path, '..', 'models/matRecognition'))
		else:
			self.model = Model(model_path = model_path)
		if mat_identify_model_path == None:
			file_path = os.path.dirname(__file__)
			self.identify_model = MatIdentification(os.path.join(file_path, '..', 'models/matIdentification'))
		else:
			self.identify_model = MatIdentification(mat_identify_model_path)
		parameters = self.model.parameters
		word_to_id, char_to_id, tag_to_id = [
		    {v: k for k, v in list(x.items())}
		    for x in [self.model.id_to_word, self.model.id_to_char, self.model.id_to_tag]
		]
		self.parameters = parameters
		self.word_to_id = word_to_id
		self.char_to_id = char_to_id
		self.tag_to_id = tag_to_id
		_, f_eval = self.model.build(training = False, **self.parameters)
		self.f_eval = f_eval
		self.model.reload()

	def mat_recognize_sent(self, input_sent):
		"""
		Recognize target/precursor in a sentence, which is a list of tokens.
		Input: list of tokens representing a sentence 
		Output: dict containing keys of precursors, targets, and other materials, 
				the value of each one is a list of index of token in the sentence
		"""
		# goal
		recognitionResult = {'precursors': [], 'targets': [], 'other_materials': []}
		# Prepare input
		words = [tmp_token['text'] for tmp_token in input_sent]
		sentence = prepare_sentence(words, self.word_to_id, self.char_to_id, \
									lower=self.parameters['lower'])
		input = create_input(sentence, self.parameters, False)
		# Prediction
		if self.parameters['crf']:
		    y_preds = np.array(self.f_eval(*input))[1:-1]
		else:
		    y_preds = self.f_eval(*input).argmax(axis = 1)
		y_preds = [self.model.id_to_tag[y_pred] for y_pred in y_preds]
		y_preds = iobes_iob(y_preds)
		mat_begin = False
		for tmp_index, y_pred in enumerate(y_preds):
			if y_pred == 'B-Pre':
				recognitionResult['precursors'].append(tmp_index)
			if y_pred == 'B-Tar':
				recognitionResult['targets'].append(tmp_index)
			if y_pred == 'B-Mat':
				recognitionResult['other_materials'].append(tmp_index)			
		return recognitionResult

	def mat_recognize(self, input_para, materials = None):
		"""
		Recognize target/precursor in a paragraph, which is plain text.
		Input: str representing a paragraph
		Output: 4 list objects of all materials, precursors, targets, and other materials 
		"""
		# goal
		mat_to_recognize = []
		precursors = []
		targets = []
		other_materials = []

		# if no materials given, use identify_model to generate default materials
		if materials == None:
			mat_to_recognize = self.identify_model.mat_identify(input_para)['LSTM']
		else:
			mat_to_recognize = materials
		# CDE tokenization
		CDE_para = CDE.doc.Paragraph(input_para)
		materials_copy = mat_to_recognize.copy()
		for tmp_sent in CDE_para:
			# prepare input sentences for LSTM	
			input_sent = []
			tag_started = False
			for t in tmp_sent.tokens:
				tmp_token = {'text': t.text, 'start': t.start, 'end': t.end}
				if tmp_token['text'].strip() == '':
					continue
				while(len(materials_copy) > 0):
					if tmp_token['start'] >= materials_copy[0]['end']:
						materials_copy.pop(0)
					else:
						break
				NER_label = 'O'
				if len(materials_copy) > 0:
					if tmp_token['start'] >= materials_copy[0]['start'] and \
						tmp_token['end'] <=  materials_copy[0]['end']:
						# beginning of a material
						if tmp_token['start'] == materials_copy[0]['start']:
							NER_label = 'B-Mat'
							tag_started = True
						elif tag_started:
							NER_label = 'I-Mat'
				if NER_label == 'O':
					input_sent.append(tmp_token)
				elif NER_label == 'B-Mat':
					input_sent.append({'text': '<MAT>', 'start': materials_copy[0]['start'], \
									'end': materials_copy[0]['end'], 'sentence': tmp_sent.text})
				else:
					pass
			recognitionResult = self.mat_recognize_sent(input_sent)
			precursors.extend([input_sent[tmp_index] for tmp_index in recognitionResult['precursors']])
			targets.extend([input_sent[tmp_index] for tmp_index in recognitionResult['targets']])
			other_materials.extend([input_sent[tmp_index] for tmp_index in recognitionResult['other_materials']])
		# reformated as the exact words in the original paragraph
		for tmp_mat in precursors + targets + other_materials:
			# debug
			# tmp_mat['text'] = 'test'
			tmp_mat['text'] = input_para[tmp_mat['start']: tmp_mat['end']]		

		return mat_to_recognize, precursors, targets, other_materials		