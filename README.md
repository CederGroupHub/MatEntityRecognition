# Materials Entity Recognition (MER)

Extract materials from a paragraph, and recognize the tartes and precursors in those materials  

Install:  

	If Git Large File Storage (lfs) is not installed on your computer, please install it fistly following the instruction on
		https://help.github.com/articles/installing-git-large-file-storage/.
	Then
		git clone git@github.com:CederGroupHub/MatEntityRecognition.git 
		cd MatEntityRecognition
		pip install -e .
	
	Spacy is used. If there is an error saying 
	    "Can't find model 'en'..." 
	It is because the spacy data is not downloaded. Please use
	    python -m spacy download en
	MaterialParser is used. Please download it from https://github.com/CederGroupHub/MaterialParser.

Use:

	from materials_entity_recognition import MatRecognition   
	model = MatRecognition()  
	all_materials, precursors, targets, other_materials = model.mat_recognize(input_para_text)  

Parameters:

	Input: plain text of a paragraph  
	Output: 4 list objects, which are all materials, precursors, targets, other materials, respectively.  

