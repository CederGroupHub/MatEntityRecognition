# Synthesis Materials Recognizer (SMR) for Materials Entity Recognition (MER)

Extract materials from a paragraph, and recognize the targets and precursors in those materials  

Install:  


	If Git Large File Storage (lfs) is not installed on your computer, please install it fistly following the instruction on
		https://help.github.com/articles/installing-git-large-file-storage/.
	Then
        git clone git@github.com:CederGroupHub/MatEntityRecognition.git 
        cd MatEntityRecognition
        pip install -e .
	
	Spacy is used. If there is an error saying: 
	    "Can't find model 'en-core-web-sm'..." 
	It is because the spacy data is not downloaded. Please use:
	    python -m spacy download en-core-web-sm
	    
    MaterialParser is used. Please find it here:
        https://github.com/CederGroupHub/MaterialParser

Use:

    # An example is in test/example.py
	from materials_entity_recognition import MatRecognition   
	model = MatRecognition()  
	result = model.mat_recognize(input_paras)  

Parameters:

	Input: list of plain text of paragraphs or plian text of a paragraph. 
	Note: input a list of paragraphs (recommended) is much faster than inputting them one by one in a loop!  
	Output: a list of (list of) dict objects, containing all materials, precursors, targets, and other materials for each sentence in the input paragraphs.  

It is also possible to use pre-defined tokens:

    # An example is in test/pre_tokens.py
    # pre_tokens is a list of list of tokens.
    # The element in the first-level list corresponds to each paragraph
    # The element in the second-level list corresponds to each sentence in each paragraph
    # Each token is dict such as {'start': 0, 'end': 4, 'text': 'text'} or 
    # an object with attributes of 'start', 'end', and 'text'. 
    result = model.mat_recognize(input_paras, pre_tokens=pre_tokens)