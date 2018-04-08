# encoding=utf8  

# chemistry related classes and methods
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import chempy
import pubchempy

import collections
import re
import regex
regex.DEFAULT_VERSION = regex.VERSION1
import os
import json
from unidecode import unidecode
import ast
import theano.tensor as T
import numpy as np
import theano
from fractions import Fraction

import .utils_database

# constant variables
# element table by symbol of elements
elementTable = {
'H': [1, 'hydrogen'],  'He': [2, 'helium'],  'Li': [3, 'lithium'],  'Be': [4, 'beryllium'],  'B': [5, 'boron'],  
'C': [6, 'carbon'],  'N': [7, 'nitrogen'],  'O': [8, 'oxygen'],  'F': [9, 'fluorine'],  'Ne': [10, 'neon'],  
'Na': [11, 'sodium'],  'Mg': [12, 'magnesium'],  'Al': [13, 'aluminium'],  'Si': [14, 'silicon'],  'P': [15, 'phosphorus'],  
'S': [16, 'sulfur'],  'Cl': [17, 'chlorine'],  'Ar': [18, 'argon'],  'K': [19, 'potassium'],  'Ca': [20, 'calcium'],  
'Sc': [21, 'scandium'],  'Ti': [22, 'titanium'],  'V': [23, 'vanadium'],  'Cr': [24, 'chromium'],  'Mn': [25, 'manganese'],  
'Fe': [26, 'iron'],  'Co': [27, 'cobalt'],  'Ni': [28, 'nickel'],  'Cu': [29, 'copper'],  'Zn': [30, 'zinc'],  
'Ga': [31, 'gallium'],  'Ge': [32, 'germanium'],  'As': [33, 'arsenic'],  'Se': [34, 'selenium'],  'Br': [35, 'bromine'],  
'Kr': [36, 'krypton'],  'Rb': [37, 'rubidium'],  'Sr': [38, 'strontium'],  'Y': [39, 'yttrium'],  'Zr': [40, 'zirconium'],  
'Nb': [41, 'niobium'],  'Mo': [42, 'molybdenum'],  'Tc': [43, 'technetium'],  'Ru': [44, 'ruthenium'],  'Rh': [45, 'rhodium'],  
'Pd': [46, 'palladium'],  'Ag': [47, 'silver'],  'Cd': [48, 'cadmium'],  'In': [49, 'indium'],  'Sn': [50, 'tin'],  
'Sb': [51, 'antimony'],  'Te': [52, 'tellurium'],  'I': [53, 'iodine'],  'Xe': [54, 'xenon'],  'Cs': [55, 'caesium'],  
'Ba': [56, 'barium'],  'La': [57, 'lanthanum'],  'Ce': [58, 'cerium'],  'Pr': [59, 'praseodymium'],  'Nd': [60, 'neodymium'],  
'Pm': [61, 'promethium'],  'Sm': [62, 'samarium'],  'Eu': [63, 'europium'],  'Gd': [64, 'gadolinium'],  'Tb': [65, 'terbium'],  
'Dy': [66, 'dysprosium'],  'Ho': [67, 'holmium'],  'Er': [68, 'erbium'],  'Tm': [69, 'thulium'],  'Yb': [70, 'ytterbium'],  
'Lu': [71, 'lutetium'],  'Hf': [72, 'hafnium'],  'Ta': [73, 'tantalum'],  'W': [74, 'tungsten'],  'Re': [75, 'rhenium'],  
'Os': [76, 'osmium'],  'Ir': [77, 'iridium'],  'Pt': [78, 'platinum'],  'Au': [79, 'gold'],  'Hg': [80, 'mercury'],  
'Tl': [81, 'thallium'],  'Pb': [82, 'lead'],  'Bi': [83, 'bismuth'],  'Po': [84, 'polonium'],  'At': [85, 'astatine'],  
'Rn': [86, 'radon'],  'Fr': [87, 'francium'],  'Ra': [88, 'radium'],  'Ac': [89, 'actinium'],  'Th': [90, 'thorium'],  
'Pa': [91, 'protactinium'],  'U': [92, 'uranium'],  'Np': [93, 'neptunium'],  'Pu': [94, 'plutonium'],  'Am': [95, 'americium'],  
'Cm': [96, 'curium'],  'Bk': [97, 'berkelium'],  'Cf': [98, 'californium'],  'Es': [99, 'einsteinium'],  'Fm': [100, 'fermium'],  
'Md': [101, 'mendelevium'],  'No': [102, 'nobelium'],  'Lr': [103, 'lawrencium'],  'Rf': [104, 'rutherfordium'],  'Db': [105, 'dubnium'],  
'Sg': [106, 'seaborgium'],  'Bh': [107, 'bohrium'],  'Hs': [108, 'hassium'],  'Mt': [109, 'meitnerium'],  'Ds': [110, 'darmstadtium'],  
'Rg': [111, 'roentgenium'],  'Cn': [112, 'copernicium'],  'Nh': [113, 'nihonium'],  'Fl': [114, 'flerovium'],  'Mc': [115, 'moscovium'],  
'Lv': [116, 'livermorium'],  'Ts': [117, 'tennessine'],  'Og': [118, 'oganesson'],  
}

# element table by name of elements
elementTable_by_name = {
'hydrogen': [1, 'H'],  'helium': [2, 'He'],  'lithium': [3, 'Li'],  'beryllium': [4, 'Be'],  'boron': [5, 'B'],  
'carbon': [6, 'C'],  'nitrogen': [7, 'N'],  'oxygen': [8, 'O'],  'fluorine': [9, 'F'],  'neon': [10, 'Ne'],  
'sodium': [11, 'Na'],  'magnesium': [12, 'Mg'],  'aluminium': [13, 'Al'],  'silicon': [14, 'Si'],  'phosphorus': [15, 'P'],  
'sulfur': [16, 'S'],  'chlorine': [17, 'Cl'],  'argon': [18, 'Ar'],  'potassium': [19, 'K'],  'calcium': [20, 'Ca'],  
'scandium': [21, 'Sc'],  'titanium': [22, 'Ti'],  'vanadium': [23, 'V'],  'chromium': [24, 'Cr'],  'manganese': [25, 'Mn'],  
'iron': [26, 'Fe'],  'cobalt': [27, 'Co'],  'nickel': [28, 'Ni'],  'copper': [29, 'Cu'],  'zinc': [30, 'Zn'],  
'gallium': [31, 'Ga'],  'germanium': [32, 'Ge'],  'arsenic': [33, 'As'],  'selenium': [34, 'Se'],  'bromine': [35, 'Br'],  
'krypton': [36, 'Kr'],  'rubidium': [37, 'Rb'],  'strontium': [38, 'Sr'],  'yttrium': [39, 'Y'],  'zirconium': [40, 'Zr'],  
'niobium': [41, 'Nb'],  'molybdenum': [42, 'Mo'],  'technetium': [43, 'Tc'],  'ruthenium': [44, 'Ru'],  'rhodium': [45, 'Rh'],  
'palladium': [46, 'Pd'],  'silver': [47, 'Ag'],  'cadmium': [48, 'Cd'],  'indium': [49, 'In'],  'tin': [50, 'Sn'],  
'antimony': [51, 'Sb'],  'tellurium': [52, 'Te'],  'iodine': [53, 'I'],  'xenon': [54, 'Xe'],  'caesium': [55, 'Cs'],  
'barium': [56, 'Ba'],  'lanthanum': [57, 'La'],  'cerium': [58, 'Ce'],  'praseodymium': [59, 'Pr'],  'neodymium': [60, 'Nd'],  
'promethium': [61, 'Pm'],  'samarium': [62, 'Sm'],  'europium': [63, 'Eu'],  'gadolinium': [64, 'Gd'],  'terbium': [65, 'Tb'],  
'dysprosium': [66, 'Dy'],  'holmium': [67, 'Ho'],  'erbium': [68, 'Er'],  'thulium': [69, 'Tm'],  'ytterbium': [70, 'Yb'],  
'lutetium': [71, 'Lu'],  'hafnium': [72, 'Hf'],  'tantalum': [73, 'Ta'],  'tungsten': [74, 'W'],  'rhenium': [75, 'b'],  
'osmium': [76, 'Os'],  'iridium': [77, 'Ir'],  'platinum': [78, 'Pt'],  'gold': [79, 'Au'],  'mercury': [80, 'Hg'],  
'thallium': [81, 'Tl'],  'lead': [82, 'Pb'],  'bismuth': [83, 'Bi'],  'polonium': [84, 'Po'],  'astatine': [85, 'At'],  
'radon': [86, 'Rn'],  'francium': [87, 'Fr'],  'radium': [88, 'Ra'],  'actinium': [89, 'Ac'],  'thorium': [90, 'Th'],  
'protactinium': [91, 'Pa'],  'uranium': [92, 'U'],  'neptunium': [93, 'Np'],  'plutonium': [94, 'Pu'],  'americium': [95, 'Am'],  
'curium': [96, 'Cm'],  'berkelium': [97, 'Bk'],  'californium': [98, 'Cf'],  'einsteinium': [99, 'Es'],  'fermium': [100, 'Fm'],  
'mendelevium': [101, 'Md'],  'nobelium': [102, 'No'],  'lawrencium': [103, 'Lr'],  'rutherfordium': [104, 'Rf'],  'dubnium': [105, 'Db'],  
'seaborgium': [106, 'Sg'],  'bohrium': [107, 'Bh'],  'hassium': [108, 'Hs'],  'meitnerium': [109, 'Mt'],  'darmstadtium': [110, 'Ds'],  
'roentgenium': [111, 'Rg'],  'copernicium': [112, 'Cn'],  'nihonium': [113, 'Nh'],  'flerovium': [114, 'Fl'],  'moscovium': [115, 'Mc'],  
'livermorium': [116, 'Lv'],  'tennessine': [117, 'Ts'],  'oganesson': [118, 'Og'],
}

elementTable_by_index = {elementTable[ele][0]: [ele, elementTable[ele][1]] for ele in elementTable}

# read chemical corpus
if os.path.exists('generated/corpusStructure.txt'):
	fr = open('generated/corpusStructure.txt', 'r', encoding='utf-8')
	chemicalCorpus = json.load(fr)
	fr.close()
else:
	chemicalCorpus = {}

# read ID database
if os.path.exists('generated/IDDatabase_MaterialsProject.json'):
	fr = open('generated/IDDatabase_MaterialsProject.json', 'r')
	IDDB = json.load(fr)
	fr.close()
else:
	IDDB = {}

# read material project ID database
if os.path.exists('rsc/allMats_name.json'):
	fr = open('rsc/allMats_name.json', 'r')
	MPDB = json.load(fr)
	fr.close()
else:
	MPDB = {}
if os.path.exists('rsc/entry_dict.json'):
	fr = open('rsc/entry_dict.json', 'r')
	MPDB_entries = json.load(fr)
	fr.close()
else:
	MPDB_entries = {}
MPDB_RCFormula = {}
for tmp_entry in MPDB_entries:
	if MPDB_entries[tmp_entry]['reduced_cell_formula'] != None:
		MPDB_RCFormula[str(utils_database.dictOrdered(MPDB_entries[tmp_entry]['reduced_cell_formula']))] = MPDB[MPDB_entries[tmp_entry]['pretty_formula']]

# classical formula detector
allElements = elementTable.keys()
allElements = sorted(allElements, key=lambda ele: len(ele), reverse=True)
allEleText = '|'.join(allElements)
# pattern_species = regex.compile(r'^[\b--[\.]](((' + allEleText + r')[\·0-9]{0,5})+)\b$')
pattern_species = regex.compile(r'^\b(((' + allEleText + r')[\·0-9]{0,5})+)\b$')
pattern_species2 = regex.compile(r'^\b(((' + allEleText + r')[\·0-9]{0,5}\w?[\·0-9]{0,5})+)\b$')
pattern_species3 = regex.compile(r'^\b(((' + allEleText + r')[\.\·0-9]{0,5}\w?[\.\·0-9]{0,5})+)\b$')
pattern_ele = regex.compile(r'((' + allEleText + r')([0-9]{0,5}))')
pattern_ele2 = regex.compile(r'(' + allEleText + r')')
pattern_singleEle = regex.compile(r'\b(' + allEleText + r')\b')
# regulate the text		
symbolToRemove = ['(', ')', '+', '-', '·', '−', '–', '.', '[', ']', '/',]
symbolToRemove = [ord(tmp_symbol) for tmp_symbol in symbolToRemove]
symbolToBlank = ['!', '"', '#', '$', '%', '&', "'", '*', ',', ':', ';', '<', '=', '>', '?', '@', '\\', '^', '_', '`', '{', '|', '}', '~']
symbolToBlank = [ord(tmp_symbol) for tmp_symbol in symbolToBlank]
substitutionTable = {tmp_symbol: None for tmp_symbol in symbolToRemove}
substitutionTable.update({tmp_symbol: ' ' for tmp_symbol in symbolToBlank})
pattern_remove = regex.compile(r'°C|\[[0-9]+\]')
pattern_subEle = regex.compile(r'[0-9]+|'+allEleText)
pattern_separator = regex.compile(r':|\*')

# Species class which contain both chemical and text information
class Species:
	# core variables
	# sentence class stored in this list
	containedInSentence = []
	# textMatched and originalText might be list in the future
	textMatched = ''
	originalText = ''

	# variables rely on only core variables
	# contain {PaperDOI: [paragraph _ids]}
	containedInParagraph = {}
	# contain paper doi
	containedInPaper = []
	timesByParagraph = collections.Counter()
	timesBySentence = collections.Counter()
	timesByWord = collections.Counter()
	relatedVerb = {}

	# variables dependent on external data
	freqByParagraph = collections.Counter()
	freqBySentence = collections.Counter()
	freqByWord = collections.Counter()
	clusterVector = {}

	def __init__(self, nameText = ''):
		self.containedInSentence = []
		self.containedInParagraph = {}
		self.containedInPaper = []
		self.textMatched = ''
		self.originalText = nameText
		self.freqByParagraph = collections.Counter()
		self.freqBySentence = collections.Counter()
		self.freqByWord = collections.Counter()
		self.timesByParagraph = collections.Counter()
		self.timesBySentence = collections.Counter()
		self.timesByWord = collections.Counter()
		self.relatedVerb = {}
		self.clusterVector = {}

	# add a new sentence while update the information
	def addSentence(self, sentence, times = 1):
		if sentence.inPaper != '':
			self.timesByWord[sentence.inPaper] += times
			if sentence.inPaper not in self.containedInPaper:
				self.containedInPaper.append(sentence.inPaper)
				self.containedInParagraph[sentence.inPaper] = []
			if sentence.inParagraph != '':
				if sentence.inParagraph not in self.containedInParagraph[sentence.inPaper]:
					self.containedInParagraph[sentence.inPaper].append(sentence.inParagraph)
					self.timesByParagraph[sentence.inPaper] += 1
		if sentence not in self.containedInSentence:
			self.containedInSentence.append(sentence)
			self.timesBySentence[sentence.inPaper] += 1
		else:
			print('Warning! This sentence has been added before!')
			print(self.nameText+'\t'+sentence.inPaper+'\t'+sentence.inParagraph+sentence.text)
			print([tmp_sent.text for tmp_sent in self.containedInSentence])

	def setFreqByParagraph(self, PaperDOI = '', freq = 0.0):
		self.freqByParagraph[PaperDOI] = freq

	def setFreqBySentence(self, PaperDOI = '', freq = 0.0):
		self.freqBySentence[PaperDOI] = freq

	def setFreqByWord(self, PaperDOI = '', freq = 0.0):
		self.freqByWord[PaperDOI] = freq

	def refreshRelatedVerb(self):
		for tmp_paper in self.containedInPaper:
			if tmp_paper not in self.relatedVerb.keys():
				self.relatedVerb[tmp_paper] = set()
		for tmp_sent in self.containedInSentence:
			self.relatedVerb[tmp_paper] |= tmp_sent.verbSet

	# checkDatabase 0 for non, 1 for IDDB, 2 for both IDDB and pubchem
	def isMaterial(self, nameText = '', mode = '', checkDatabase = 0, guessLevel = 0):
		result = False
		if nameText != '':
			materialName = nameText
		else:
			materialName = self.originalText
		if materialName == '':
			print('Error! No originalText found for material determination!')
			result = False
			return result
# 		# remove or substitute the delimiter symbols
		tmp_text = pattern_remove.sub('', materialName)
		tmp_text = tmp_text.translate(substitutionTable)
		if mode == 'test':
			print('To be evaluated isMaterial():\t'+tmp_text)
		# if materialName == 'CR2032':
		# 	print('here')
		if pattern_species.match(tmp_text):
			# if materialName == 'CR2032':
			# 	print('here2')
			if materialName == 'Co1−xS':
				print(tmp_num)
			result = True
			try:
				tmp_mat = chempy.Substance.from_formula(materialName)
				# if materialName == 'CR2032':
				# 	print(tmp_mat.composition)
				for (tmp_ele, tmp_num) in tmp_mat.composition.items():
					if tmp_num > 100:
						result = False	
			except Exception as e:
				pass
		elif pattern_species2.match(tmp_text):
			result = True
			tmp_eles = [tmp_ele[1] for tmp_ele in pattern_ele.findall(materialName)]
			tmp_eles = set(tmp_eles)	
			if len(tmp_eles) < 2:
				result = False
			tmp_text2 = pattern_remove.sub('', materialName)
			tmp_text2 = tmp_text2.translate({ord('.'): None})
			tmp_nums = regex.split(allEleText+'|\(|\)|·|\[|\]', tmp_text2)
			tmp_sub = ''
			for tmp_num in tmp_nums:
				if tmp_num == '' or tmp_num == None:
					continue
				# if materialName == 'Co1−xS':
				# 	print(tmp_num)
				if not regex.match('^[0-9]{0,2}[+-−–]?[\w--[0-9]]?([0-9]{0,2}[+-−–])?$', tmp_num):
					result = False
				else:
					tmp_sub += tmp_num
			tmp_sub = set(list(tmp_sub))
			if len(list(filter(lambda tmp_char: (tmp_char >= 'A' and tmp_char <= 'Z'), tmp_sub))) > 1  \
			or len(list(filter(lambda tmp_char: (tmp_char >= 'a' and tmp_char <= 'z'), tmp_sub))) > 3:
				 result = False

		elif tmp_text.lower() in elementTable_by_name.keys():
			result = True
		else:
			# if materialName == 'CR2032':
			# 	print('here4')			
			tmp_m = pattern_ele.findall(tmp_text)
			if len(tmp_m) > 0:
				tmp_text = pattern_subEle.sub('', tmp_text)
				# what if TMBaO3 used?
				tmp_text = set(tmp_text)	
				if len(tmp_text) > 4:
					result = False
					return result
				else:
					tmp_upper = set(filter(lambda tmp_char: tmp_char >= 'A' and tmp_char <= 'Z', tmp_text))
					tmp_lower = tmp_text - tmp_upper
					if len(tmp_upper) == 0:
						result = False
						return result
					# if materialName == 'MNb2O6':
					# 	print([tmp_sent.text for tmp_sent in self.containedInSentence])
					if len(self.containedInSentence) > 0:
						matched_upper = set()
						for tmp_sent in self.containedInSentence:
							tmp_pos = tmp_sent.text.find(materialName)
							tmp_m2 = regex.findall('(\([^\(\)]+\))', tmp_sent.text[tmp_pos:])
							if len(tmp_m2) > 0:
								# find in parenthesis (E.g. M=Ti, Fe)
								for tmp_note in tmp_m2:
									tmp_note2 = pattern_remove.sub('', tmp_note)
									tmp_note2 = tmp_note2.translate(substitutionTable)
									if len(pattern_singleEle.findall(tmp_note2)) > 0:
										tmp_words = tmp_note2.split()
										for tmp_char in tmp_upper:
											if tmp_char in tmp_words:
												matched_upper.update(tmp_char)
								if matched_upper == tmp_upper:
									# print('text:\t', tmp_m2, 'matched symbol', matched_upper)
									result = True
							if result != True:							
								# find in the whole sentence
								tmp_note = pattern_remove.sub('', tmp_sent.text)
								tmp_note = tmp_note.translate(substitutionTable)
								if len(pattern_singleEle.findall(tmp_note)) > 0:
									tmp_words = tmp_note.split()
									for tmp_char in tmp_upper:
										if tmp_char in tmp_words:
											matched_upper.update(tmp_char)
								if matched_upper == tmp_upper:
									if guessLevel > 0:
										print('Warning! The word ' + materialName + ' might be not a material! guessLevel:\y' + str(guessLevel))
										result = True
									else:
										result = False
					if result != True:
						# if representative symbol used, the number of other elements should be all ones
						if len(list(filter(lambda tmp_ele: tmp_ele[2] != '', tmp_m))) == 0:
							result = False
							return result
						for tmp_ele in tmp_m:
							# the number of a center element should not be too large
							if tmp_ele[2] != '':
								if int(tmp_ele[2]) > 99:
									result = False
									return result
						if len(tmp_upper) == 1:
							if len(tmp_lower) == 0:
								result = True
							elif len(tmp_lower) == 1:
								if guessLevel > 0:
									print('Warning! The word ' + materialName + ' might be not a material! guessLevel:\y' + str(guessLevel))
									result = True
								else:
									result = False
							else:
								result = False
						elif len(tmp_upper) == 2:
							if len(tmp_lower) <= 1:
								if guessLevel > 0:
									print('Warning! The word ' + materialName + ' might be not a material! guessLevel:\y' + str(guessLevel))
									result = True
								else:
									result = False
							else:
								result = False	
						else:
							result = False
				if result == True and len(tmp_upper) > 0:
					tmp_materialName = list(materialName)
					for tmp_ele in tmp_upper:
						tmp_len = len(tmp_materialName)
						for i in range(tmp_len):
							if materialName[i] == tmp_ele:
								if materialName[i] not in elementTable.keys():
									if (i+1) < tmp_len:
										if materialName[i:i+2] not in elementTable.keys():
											tmp_materialName[i] = 'H'
									else:
										tmp_materialName[i] = 'H'
								else:
									print('Error! There should be upper letters same as a element name now!')
					result = self.isMaterial(nameText = ''.join(tmp_materialName))
			elif checkDatabase > 0:
				if self.findID(materialName) != None:
					result = True
				elif checkDatabase > 1:
					try:
						tmp_search = pubchempy.get_compounds(materialName.lower(), 'name')
						if len(tmp_search) > 0:
							result = True
						else:
							result = False
					except Exception as e:
						print('Warning! Network unavailable for PubChem!')
						result = False
		return result

	def isMaterial2(self, nameText = '', mode=''):
		result = False
		words = self.originalText.split()
		if mode == 'test':
			print('To be evaluated isMaterial2():\t'+self.originalText)		
		if len(words) == 0:
			print('Error! No originalText found for material determination!')
		elif len(words) == 1:
			# need to add a parameter for function isMaterial
			result = self.isMaterial(nameText = words[0], checkDatabase = 2)
		else:
			coreWords = []
			if self.originalText == 'oxide':
				print('here2 in isMaterial2()')
			for tmp_index, tmp_word in enumerate(words):
				if tmp_word in elementTable.keys() or tmp_word.lower() in elementTable_by_name.keys():
					words[tmp_index] = '$element$'
				elif self.isMaterial(nameText = tmp_word, checkDatabase = 1):
					words[tmp_index] = '$compound$'
				else:
					tmp_word_unstem = WordNetLemmatizer().lemmatize(tmp_word.lower(), 'n') 
					tmp_word_unstem = pattern_remove.sub('', tmp_word_unstem)
					if tmp_word_unstem in chemicalCorpus.keys():
						words[tmp_index] = tmp_word_unstem
						coreWords.append(words[tmp_index])
			tmp_text = ' '.join(words)	
			for tmp_word in coreWords:
				if tmp_text in chemicalCorpus[tmp_word][0]:
					result = True
					break
			if result != True:
				if self.findID(self.originalText) != None:
					result = True
				else:				
					tmp_search = pubchempy.get_compounds(self.originalText.lower(), 'name')
					if len(tmp_search) > 0:
						result = True
		return result

	def resolveFormula(self, nameText=None):
		formula = None
		if nameText != None:
			materialName = nameText
		else:
			materialName = self.originalText

		# deal with special non-ASCII character
		materialName = unidecode(materialName)

		# replace all [] with ()?
		
 		# deal with separators, and convert separators into the same symbol like |
		# deal with |
		# need to be modified for the cases like Mg(OH)2·4MgCO3·5H2O
		materialName = pattern_separator.sub('|', materialName)
		index_star = materialName.find('|')
		if index_star > -1:
			tmp_m = regex.match(' *([\.\+\- 0-9a-z]+).*', materialName[index_star+1:])
			if tmp_m:
				star_num = tmp_m.group(1)
				star_cont = index_star + 1 + materialName[index_star+1:].find(star_num) + len(star_num)
				# if paren_num = 1-x use theano
				# paren_num = float(paren_num)
				star_num = self.resolveEleNumber(star_num)
			else:
				star_num = 1
				star_cont = index_star + 1

			left_formula = self.resolveFormula(nameText = materialName[0:index_star])
			if materialName[0:index_star] != '' and (left_formula == None or len(left_formula) == 0):
				# print('Error! left_formula cannot be resolved!')
				return None	
			right_formula = self.resolveFormula(nameText =  materialName[star_cont:])
			if materialName[star_cont:] != '' and (right_formula == None or len(left_formula) == 0):
				# print('Error! right_formula cannot be resolved!')
				return None
			for tmp_ele in right_formula:
				tmp_ele[1] *= star_num
			return left_formula + right_formula


		# deal with parentheses
		left_parens = regex.findall('\(', materialName)
		# need to be modified right_parens should be the first one matching the first left_parens, a heap should be used to found that
		right_parens = regex.findall('\)', materialName)
		if len(left_parens) != len(right_parens):
			print('Warning! len(left_parens) != len(right_parens)!')
			return None
			if len(left_parens) > len(right_parens):
				materialName = materialName.replace('(', '', len(left_parens)-len(right_parens))
			else:
				materialName = materialName[::-1].replace(')', '', len(right_parens)-len(left_parens))
				materialName = materialName[::-1]
		# how to deal with xA2B3 + yA3B2?
		# add specification to deal with the separators here

		paren_start = materialName.find('(')
		paren_end= -1
		if paren_start > -1:
			paren_stack = []
			for (i, c) in enumerate(materialName[paren_start:]):
				if c == '(':
					paren_stack.append('(')
				elif c == ')':
					paren_stack.pop(-1)
				if len(paren_stack) == 0:
					paren_end = paren_start + i
					break
			tmp_m = regex.match(' *([\.\+\- 0-9a-z]+).*', materialName[paren_end+1:])
			if tmp_m:
				paren_num = tmp_m.group(1)
				parent_cont = paren_end + 1 + materialName[paren_end+1:].find(paren_num) + len(paren_num)
				# if paren_num = 1-x use theano
				# paren_num = float(paren_num)
				paren_num = self.resolveEleNumber(paren_num)
				if paren_num == None:
					# print('Error! number of element (paren_num) cannot be resolved!')
					return None
			else:
				paren_num = 1
				parent_cont = paren_end + 1

			left_formula = self.resolveFormula(nameText = materialName[0:paren_start])
			if materialName[0:paren_start] != '' and (left_formula == None or len(left_formula) == 0):
				# print('Error! left_formula cannot be resolved!')
				return None
			center_formula = self.resolveFormula(nameText = materialName[paren_start+1:paren_end])
			if materialName[paren_start+1:paren_end] != '' and (center_formula == None or len(center_formula) == 0):
				# print('Error! center_formula cannot be resolved!')
				return None
			for tmp_ele in center_formula:
				tmp_ele[1] *= paren_num


			right_formula = self.resolveFormula(nameText = materialName[parent_cont:])
			if materialName[parent_cont:] != '' and (right_formula == None or len(right_formula) == 0):
				# print('Error! right_formula cannot be resolved!')
				return None
			return left_formula + center_formula + right_formula 

		# find elements
		# need to be modified to deal with cases like LATP
		ele_pairs = []
		all_eles = regex.finditer(pattern_ele2, materialName)
		all_eles = list(all_eles)
		for (i, tmp_ele) in enumerate(all_eles):
			# if paren_num = 1-x use theano
# 			check if there are unkonwn symbols or numbers before the first element
			if i == 0 and tmp_ele.start() != 0:
				if self.resolveEleNumber(materialName[0:tmp_ele.start()]) == None:
					ele_pairs = None
					break
				
			if i < (len(all_eles)-1):
				if regex.match('[\.\+\- 0-9a-z]+', materialName[tmp_ele.end(): all_eles[i+1].start()]):
					ele_num = materialName[tmp_ele.end(): all_eles[i+1].start()]
				elif tmp_ele.end() == all_eles[i+1].start():
					ele_num = '1'
				else:
					# print('Error! number of element cannot be resolved!')
					ele_pairs = None
					break
					# return None
			else:
				if regex.match('[\.\+\- 0-9a-z]+', materialName[tmp_ele.end():]):
					ele_num = materialName[tmp_ele.end():]
				elif materialName[tmp_ele.end():] == '':
					ele_num = '1'
				else:
					# print('Error! num of element cannot be resolved!')
					ele_pairs = None
					break
					# return None	
			#if paren_num = 1-x use theano
			# ele_num = float(ele_num)
			ele_num = self.resolveEleNumber(ele_num)
			if ele_num == None:
				# print('Error! number of element (ele_num) cannot be resolved!')
				ele_pairs = None
				break
				# return None
			ele_pairs.append([tmp_ele.group(), ele_num])

		if ele_pairs == None or len(ele_pairs) == 0:
			additional_info = ''
			tmp_sent = self.containedInSentence[0]
			tmp_pos = tmp_sent.find(materialName)
			tmp_m = regex.match('[^\(]*\(([^\(\)]+)\).*', tmp_sent[tmp_pos+len(materialName):])
			if tmp_m:
				additional_info = unidecode(tmp_m.group(1)) 
			if additional_info != '':
				tmp_m = regex.match(' *([A-Z][a-z]?) *=.*', additional_info)
				if tmp_m:
					ele_symbol = tmp_m.group(1)
					ele_subed = regex.finditer(pattern_ele2, additional_info)
					ele_subed = [tmp_ele.group() for tmp_ele in ele_subed]						
					if materialName.find(ele_symbol) > -1 and len(ele_subed) > 0:
						ele_pairs = []
						for tmp_index, tmp_ele in enumerate(ele_subed):
							if ele_symbol == tmp_ele:
								print('Error! ele_symbol == tmp_ele')
								continue
							materialName_subbed = materialName.replace(ele_symbol, tmp_ele)
							tmp_ele_pairs = self.resolveFormula(nameText = materialName_subbed)
							for i, tmp_ele_pair in enumerate(tmp_ele_pairs):
								if tmp_index == 0:	
									if tmp_ele_pair[0] != tmp_ele:
										ele_pairs.append(tmp_ele_pair)		
									else:
										ele_pairs.append([ele_symbol, tmp_ele_pair[1], [tmp_ele]])										
								else:
									if tmp_ele_pair[0] != tmp_ele:
										if ele_pairs[i][0] != tmp_ele_pair[0]:
											print('Error! ele_pairs[i][0] != tmp_ele_pair[0]')
									else:
										ele_pairs[i][2].append(tmp_ele)

		return ele_pairs

	def resolveEleNumber(self, numText):
		number = None
		try:
			number = float(numText)
		except Exception as e:
			pass
		if number == None:
			try:
				number = float(Fraction(numText))
			except Exception as e:
				pass
		if number == None:
			_local = locals()
			variables = []
			try:
				ast_tree = ast.parse(numText.replace(' ', ''))
				for node in ast.walk(ast_tree):
					if isinstance(node, ast.Name):
						if len(node.id) < 2:
							variables.append(node.id)	
						else:
							variables = []
							break	
			except Exception as e:
				pass
			if len(variables) == 0:
				# ion
				if regex.match('[0-9]*[\+\-]', numText):
					number = 1.0
				else:
					number = None
			elif len(variables) == 1:
				exec(', '.join(variables) + '=' + 'T.dscalar('+str(variables).strip('[]')+')', globals(), _local) 
				exec('number = ' + numText, globals(), _local)	
				number = _local['number']
			else:
				exec(', '.join(variables) + '=' + 'T.dscalars('+str(variables).strip('[]')+')', globals(), _local) 
				exec('number = ' + numText, globals(), _local)	
				number = _local['number']
			if isinstance(number, tuple):
				number = number[0]
		return number		

	def findID(self, nameText = '', checkCID = False):
		ID = None
		if nameText != '':
			materialName = nameText
		else:
			materialName = self.originalText
		tmp_words = materialName.split()
		nameLower = ' '.join([tmp_word[0].lower() + tmp_word[1:] for tmp_word in tmp_words])
		nameUpper = ' '.join([tmp_word[0].upper() + tmp_word[1:] for tmp_word in tmp_words])
		if materialName in IDDB:
			ID = IDDB[materialName]
		elif nameLower in IDDB:
			ID = IDDB[nameLower]
		elif nameUpper in IDDB:
			ID = IDDB[nameUpper]
		else:
			ID = None

		if ID != None and checkCID == True:
			if len(ID[1]) == 0:
				cidFound = False	
				for tmp_sid in ID[0]:
					tmp_substance = pubchempy.Substance.from_sid(int(tmp_sid))
					try:
						if tmp_substance.standardized_cid != None:
							cidFound = True
					except Exception as e:
						cidFound = False
				if cidFound == False:
					ID = None
		return ID


	def findMPID(self, nameText = ''):
		ID = None		
		resolveFormulaUsed = False	
		if nameText != '':
			materialName = nameText
		else:
			materialName = self.originalText
		tmp_words = materialName.split()
		nameFirstUpper = ' '.join([tmp_words[0][0].upper() + tmp_words[0][1:]] + tmp_words[1:])
		nameLower = ' '.join([tmp_word[0].lower() + tmp_word[1:] for tmp_word in tmp_words])
		nameUpper = ' '.join([tmp_word[0].upper() + tmp_word[1:] for tmp_word in tmp_words])
		nameContract = ''.join(tmp_words)
		if materialName in MPDB:
			ID = MPDB[materialName]
		elif nameFirstUpper in MPDB:
			ID = MPDB[nameFirstUpper]
		elif nameLower in MPDB:
			ID = MPDB[nameLower]
		elif nameUpper in MPDB:
			ID = MPDB[nameUpper]
		elif nameContract in MPDB:
			ID = MPDB[nameContract]
		else:
			try:
				resolveFormulaUsed = True
				tmp_mat_eles = self.resolveFormula(materialName)
				tmp_mat = {}
				all_vars = set()
				for tmp_ele in tmp_mat_eles:
					if tmp_ele[0] not in tmp_mat:
						tmp_mat[tmp_ele[0]] = 0
					if isinstance(tmp_ele[1], float):
						tmp_mat[tmp_ele[0]] += tmp_ele[1]
					else:
						var_set = utils_database.getAllVar(tmp_ele[1])
						all_vars.update({tmp_var.name for tmp_var in var_set})
						# var_values = {tmp_var:1/3 for tmp_var in var_set}
						# tmp_mat[tmp_ele[0]] += tmp_ele[1].eval(var_values)
						tmp_mat[tmp_ele[0]] += tmp_ele[1]

				if len(all_vars) == 0:
					RCFormula = utils_database.dictOrdered(tmp_mat)
					if str(RCFormula) in MPDB_RCFormula:
						ID = MPDB_RCFormula[str(RCFormula)]
						# print('here', materialName, RCFormula, ID)
				else:
					# print(materialName)
					para_trial = {}
					ID = []
					for tmp_var in all_vars:
						para_trial[tmp_var] = np.arange(0,1.1,0.1)
					para_trial = utils_database.getAllParaCombo(para_trial)
					for tmp_para in para_trial:
						tmp_mat2 = {}
						for tmp_ele in tmp_mat:
							if not isinstance(tmp_mat[tmp_ele], float):
								var_set = utils_database.getAllVar(tmp_mat[tmp_ele])
								var_dict = {tmp_var: round(tmp_para[tmp_var.name], 1) for tmp_var in var_set}
								tmp_mat2[tmp_ele] = round(float(tmp_mat[tmp_ele].eval(var_dict)), 1)
							else:
								tmp_mat2[tmp_ele] = tmp_mat[tmp_ele]
							if abs(tmp_mat2[tmp_ele]) < 0.001:
								del tmp_mat2[tmp_ele]								
						RCFormula = utils_database.dictOrdered(tmp_mat2)
						# print(RCFormula)
						if str(RCFormula) in MPDB_RCFormula:
							ID.extend(MPDB_RCFormula[str(RCFormula)]) 
							# print('here', materialName, RCFormula, ID[-1])
					if len(ID) == 0:
						ID = None	

			# 	# # use chempy from formula									
			# 	# tmp_m = pattern_species3.match(materialName)
			# 	# if tmp_m:
			# 	# 	all_eles = [ele[1] for ele in pattern_ele.findall(materialName)]
			# 	# 	all_eles = set(all_eles)
			# 	# else:
			# 	# 	all_eles = set()	
			# 	# tmp_mat = chempy.Substance.from_formula(materialName)
			# 	# if len(tmp_mat.composition) == len(all_eles):
			# 	# 	RCFormula = utils_database.dictOrdered({elementTable_by_index[ele][0]: float(tmp_mat.composition[ele]) for ele in tmp_mat.composition})
			# 	# 	if str(RCFormula) in MPDB_RCFormula:
			# 	# 		ID = MPDB_RCFormula[str(RCFormula)]
			# 	# 		print('here', materialName, RCFormula, ID)
			except Exception as e:
				ID = None
				resolveFormulaUsed = False					
		return ID, resolveFormulaUsed

# # get possibleEquation between precursors and targets
# # precursors and targets are dict
# # {matName: set(elements)}
# def possibleEquation(precursors, targets):
# 	eqs = []
# 	for tmp_target in targets:
# 		precursor_candidate = []
# 		all_pre_eles = set()
# 		for tmp_precursor in precursors:
# 			if len(precursors[tmp_precursor] & targets[tmp_target] - {'O'}) > 0:
# 				precursor_candidate.append(tmp_precursor)
# 				all_pre_eles.update(precursors[tmp_precursor])
# 		if targets[tmp_target].issubset(all_pre_eles):

# 			eqs.append([precursor_candidate, tmp_target])
# 	return eqs	

# get possibleEquation between precursors and targets
# precursors and targets are dict
# {matName: set(elements)}
def possibleEquation(precursors, targets):
	eqs = []
	for tmp_target in targets:
		precursor_candidate = []
		all_pre_eles = set()
		for tmp_precursor in precursors:
			if len(precursors[tmp_precursor] & targets[tmp_target] - {'O'}) > 0:
				precursor_candidate.append((tmp_precursor, precursors[tmp_precursor]))
				all_pre_eles.update(precursors[tmp_precursor])
		eqs.extend(possibleEquation2(precursor_candidate, (tmp_target, targets[tmp_target])))
	return eqs	


# get possibleEquation between precursors and targets
# precursors list of tuples and target is tuple
# [(matName: set(elements))]
def possibleEquation2(precursors, target):
	eqs = []
	all_pre_eles = set()
	for tmp_precursor in precursors:
		all_pre_eles.update(tmp_precursor[1])
	if target[1].issubset(all_pre_eles):
		needAllPres = True
		for i, tmp_precursor in enumerate(precursors):
			precursors_copy = precursors.copy()
			precursors_copy.pop(i)
			tmp_eqs = possibleEquation2(precursors_copy, target)
			if len(tmp_eqs) > 0:
				needAllPres = False
				tmp_eqs_str = set([str(tmp_eq) for tmp_eq in eqs])
				for tmp_eq in tmp_eqs:
					if str(tmp_eq) in tmp_eqs_str:
						continue
					else:
						eqs.append(tmp_eq)
						tmp_eqs_str.add(str(tmp_eq))
		if needAllPres:
			# banlance reaction
			balanced = False
			reac = set([tmp_mat[0] for tmp_mat in precursors])
			prod = set([target[0]])
			try:
				left, right = chempy.balance_stoichiometry(reac, prod)
				if str(list(left.values())[0]) == 'nan':
					balanced = False
				else:
					balanced = True
			except Exception as e:
				balanced = False			
			if not balanced:
				try:
					prod = set([target[0], 'O'])
					left, right = chempy.balance_stoichiometry(reac, prod)
					if str(list(left.values())[0]) == 'nan':
						balanced = False
					else:
						balanced = True
				except Exception as e:
					balanced = False
			if not balanced:
				if 'H' in all_pre_eles:
					try:
						prod = set([target[0], 'O', 'H'])
						left, right = chempy.balance_stoichiometry(reac, prod)
						if str(list(left.values())[0]) == 'nan':
							balanced = False
						else:
							balanced = True
					except Exception as e:
						balanced = False
			if not balanced:
				if 'C' in all_pre_eles:
					try:
						prod = set([target[0], 'O', 'C'])
						left, right = chempy.balance_stoichiometry(reac, prod)
						if str(list(left.values())[0]) == 'nan':
							balanced = False
						else:
							balanced = True
					except Exception as e:
						balanced = False
			if not balanced:
				if 'C' in all_pre_eles and 'H' in all_pre_eles:
					try:
						prod = set([target[0], 'O', 'C', 'H'])
						left, right = chempy.balance_stoichiometry(reac, prod)
						if str(list(left.values())[0]) == 'nan':
							balanced = False
						else:
							balanced = True
					except Exception as e:
						balanced = False	
			if not balanced:
				if 'Cl' in all_pre_eles:
					try:
						prod = set([target[0], 'Cl'])
						left, right = chempy.balance_stoichiometry(reac, prod)
						if str(list(left.values())[0]) == 'nan':
							balanced = False
						else:
							balanced = True
					except Exception as e:
						balanced = False											
			if balanced:
				eqs.append([left, right])
	return eqs	

def equivalentMaterials(mat_text_1, mat_text_2):
	tmp_species = Species()
# 	try to resolve mat 1
	tmp_species.originalText = mat_text_1
	elements = {}
	# try to find elements through database
	ID, resolveFormulaUsed = tmp_species.findMPID()
	if ID != None and resolveFormulaUsed != True:
		for tmp_id in ID:
			elements = MPDB_entries[tmp_id]['reduced_cell_formula']
  	
	# try to resolve formula
	if elements== None or len(elements) == 0:
		try:
			elements = tmp_species.resolveFormula()
			# print(elements)
		except Exception as e:
			# print('cannot resolve formula: ', tmp_mat_text)
			pass
	if elements== None or len(elements) == 0:
		return False
	else:
		if elements.__class__.__name__ == 'list':
			elements = {e[0]: e[1:] for e in elements}
		else:
			elements = {k: [elements[k]] for k in elements}
		elements_1 = elements
# 	try to resolve mat 1
	tmp_species.originalText = mat_text_2
	elements = {}
	# try to find elements through database
	ID, resolveFormulaUsed = tmp_species.findMPID()
	if ID != None  and resolveFormulaUsed != True:
		for tmp_id in ID:
			elements = MPDB_entries[tmp_id]['reduced_cell_formula']
  	
	# try to resolve formula
	if elements== None or len(elements) == 0:
		try:
			elements = tmp_species.resolveFormula()
			# print(elements)
		except Exception as e:
			# print('cannot resolve formula: ', tmp_mat_text)
			pass
	if elements== None or len(elements) == 0:
		return False	
	else:
		if elements.__class__.__name__ == 'list':
			elements = {e[0]: e[1:] for e in elements}
		else:
			elements = {k: [elements[k]] for k in elements}

		elements_2 = elements
		
	equivalent = True
	if len(elements_1) != len(elements_2):
		equivalent = False
	for ele in elements_1:
		if ele in elements_2:
			if elements_1[ele] != elements_2[ele]:
				equivalent = False
				break
		else:
			equivalent = False
			break
	return equivalent
