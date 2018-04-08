# Materials Entity Recognition (MER)

Extract materials from a paragraph, and recognize the tartes and precursors in those materials  

Use:
	from materials_entity_recognition import MatRecognition
	model = MatRecognition()
	all_materials, precursors, targets, other_materials = model.mat_recognize(input_para_text)

Parameters:
	Input: plain text of a paragraph
	Output: 4 list objects, which are all materials, precursors, targets, other materials, respectively.

