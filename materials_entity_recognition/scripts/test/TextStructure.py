# encoding=utf8  

# class about text structure to compensate the structure in current synthesis paper database

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import chemdataextractor as CDE
from unidecode import unidecode

import re
import regex
import collections

import chemistry

# regulate the text
symbolToRemove = ['(', ')', '+', '-', '·', '−']
symbolToRemove = [ord(tmp_symbol) for tmp_symbol in symbolToRemove]
symbolToBlank = ['!', '"', '#', '$', '%', '&', "'", '*', ',', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
symbolToBlank = [ord(tmp_symbol) for tmp_symbol in symbolToBlank]
substitutionTable = {tmp_symbol: None for tmp_symbol in symbolToRemove}
substitutionTable.update({tmp_symbol: ' ' for tmp_symbol in symbolToBlank})
pattern_remove = re.compile(r'°C')

class Sentence:
	# doi used as id of paper
	inPaper = ''
	# paragraph _id used
	inParagraph = ''
	text = ''
	tags = []
	verbSet = set()
	stemSet = set()

	def __init__(self, PaperDOI = '', ParagraphID = '', fullText = ''):
		self.inPaper = PaperDOI
		self.inParagraph = ParagraphID
		self.text = fullText
		if fullText != '':
			tmp_tokens = nltk.word_tokenize(self.text)
			self.tags = nltk.pos_tag(tmp_tokens)
			tmp_verbs = list(filter(lambda tmp_tag: tmp_tag[1].startswith('VB'), self.tags))
			tmp_verbs = [WordNetLemmatizer().lemmatize(tmp_verb[0], 'v') for tmp_verb in tmp_verbs]
			self.verbSet = set(tmp_verbs)
			self.stemSet = set([PorterStemmer().stem(tmp_tag[0]) for tmp_tag in self.tags])
		else:
			self.tags = []
			self.verbSet = set()
			self.stemSet = set()

	def indexInRegulated(self, word):
		# remove or substitute the delimiter symbols
		tmp_text = self.text.translate(substitutionTable)
		tmp_text = pattern_remove.sub(' ', tmp_text)
		if  word+' ' in tmp_text:
			return tmp_text.index(word+' ')
		elif word+'\b' in tmp_text:
			return tmp_text.index(word+'\b')
		else:
			return tmp_text.index(word)

	def indexVerbInRegulated(self, word):
		tmp_verbs = list(filter(lambda tmp_tag: tmp_tag[1].startswith('VB'), self.tags))
		tmp2_verbs = [WordNetLemmatizer().lemmatize(tmp_verb[0], 'v') for tmp_verb in tmp_verbs]
		return self.indexInRegulated(tmp_verbs[tmp2_verbs.index(word)][0])

	def indexStemInRegulated(self, word):
		tmp_stem = [PorterStemmer().stem(tmp_tag[0]) for tmp_tag in self.tags]
		return self.indexInRegulated(self.tags[tmp_stem.index(word)][0])

	def isVerbType(self, type, word):
		tmp_verbs = list(filter(lambda tmp_tag: tmp_tag[1].startswith('VB'), self.tags))
		tmp2_verbs = [WordNetLemmatizer().lemmatize(tmp_verb[0], 'v') for tmp_verb in tmp_verbs]
		return tmp_verbs[tmp2_verbs.index(word)][1] == type

	def materialsSearch(self, useCDE = True, filterCDE = False):
		# target:
		species = set()
		result = {}

		# tmporary variables
		tmp_mat = chemistry.Species()
		tmp_mat.addSentence(self)

		# # detect materials by each word with more than 2 elements			
		# remove or substitute the delimiter symbols
		tmp_words = self.text.split()
		for i in range(len(tmp_words)):
			tmp_words[i] = tmp_words[i].strip(',.;?! ')
			if len(tmp_words[i]) > 0:
				if tmp_words[i][0] == '(' and tmp_words[i][-1] == ')':
					tmp_words[i] = tmp_words[i].strip('\(\)')
		# tmp_words = [tmp_word.strip('\(\),.;?! ') for tmp_word in tmp_words]
		# tmp_words = tmp_CDESent.tokens
		# tmp_words = [tmp_token.text for tmp_token in tmp_words]
		for tmp_word in tmp_words:
			tmp_eles = [tmp_ele[1] for tmp_ele in chemistry.pattern_ele.findall(tmp_word)]
			tmp_eles = set(tmp_eles)
			if len(tmp_eles) > 1:
				if tmp_mat.isMaterial(nameText = tmp_word, checkDatabase = 0, guessLevel = 0):					
					species.add(tmp_word)

		# detect materials with chemdataextractor, if no filter
		if useCDE == True:
			tmp_doc = CDE.Document(self.text)
			tmp_matTexts = [tmp_cem.text for tmp_cem in tmp_doc.cems]
			tmp_matTexts = collections.Counter(tmp_matTexts)
			for tmp_matText in tmp_matTexts:
				tmp_mat.originalText = tmp_matText
				if filterCDE == True:
					if tmp_mat.isMaterial2():
						species.add(tmp_matText)
				else:
					species.add(tmp_matText)

		# get position
		for tmp_species in species:
			tmp_locations = self.materialLocation(tmp_species)
			if len(tmp_locations) > 0:
				result[tmp_species] = tmp_locations
		return result

	def materialLocation(self, matName):
		position = []
		tmp_pos = self.text.find(matName)
		tmp_len = len(matName)
		while(tmp_pos > -1):
			tmp_individual = True
			if (tmp_pos+tmp_len) < len(self.text):
				if regex.match('[\-0-9a-zA-Z]', unidecode(self.text[tmp_pos+tmp_len])):
					tmp_individual = False
			if (tmp_pos-1) > 0:
				if regex.match('[\-0-9a-zA-Z]', unidecode(self.text[tmp_pos-1])):
					tmp_individual = False					
			if tmp_individual == True:
				position.append((tmp_pos, tmp_pos+tmp_len))
			tmp_pos2 = self.text[tmp_pos+tmp_len:].find(matName)
			if tmp_pos2 > -1:
				tmp_pos += tmp_len + tmp_pos2
			else:
				tmp_pos = -1
		return position			

# this function is used to combine tokens, which represent materials
# dedicated to process annotated tokens
# return a list of material strings
def combineTokens(tokens, paraText):
	materials = []
	sortedTokens = sorted(tokens, key=lambda t: t['start'])
	for i, tmp_token in enumerate(sortedTokens):
		if len(materials) == 0:
			materials.append(tmp_token['text'])
			continue
		if paraText[sortedTokens[i-1]['end']: tmp_token['start']].strip() == '':
			materials[-1] = materials[-1] + paraText[sortedTokens[i-1]['end']: tmp_token['end']]
		else:
			materials.append(tmp_token['text'])
	materials = list(set(materials))
	return materials

# this function is used to combine tokens, which represent materials
# dedicated to process annotated tokens
# return a list of material tokens
# text is the text in original plain text
# tokenText is the token text concatenated with blank
def combineTokens2(tokens, paraText):
	materials = []
	sortedTokens = sorted(tokens, key=lambda t: t['start'])
	for i, tmp_token in enumerate(sortedTokens):
		if len(materials) == 0:
			materials.append({'text': tmp_token['text'], 'start': tmp_token['start'], 'end': tmp_token['end'], 'tokenText': tmp_token['text']})
			continue
		if paraText[sortedTokens[i-1]['end']: tmp_token['start']].strip() == '':
			materials[-1]['end'] = tmp_token['end']
			materials[-1]['text'] = paraText[materials[-1]['start']: materials[-1]['end']]
			materials[-1]['tokenText'] = materials[-1]['tokenText'] + ' ' + tmp_token['text']  
		else:
			materials.append({'text': tmp_token['text'], 'start': tmp_token['start'], 'end': tmp_token['end'], 'tokenText': tmp_token['text']})
	return materials
