import os
from nltk.parse import stanford
from nltk.tree import Tree
from unidecode import unidecode
import nltk
from nltk.stem import WordNetLemmatizer

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# common variables
file_path = os.path.dirname(__file__)
stanford_parser_folder = os.path.join(file_path, '..', 'models/stanfordParser')
stanford_model_path = os.path.join(file_path, '..', 'models/stanfordParser/englishPCFG.ser.gz') 
os.environ['STANFORD_PARSER'] = stanford_parser_folder
os.environ['STANFORD_MODELS'] = stanford_parser_folder
tree_parser = stanford.StanfordParser(model_path=stanford_model_path)
dep_parser = stanford.StanfordDependencyParser(model_path=stanford_model_path)

tabu_IN_words = set(['of'])
nltk_lemmatizer = WordNetLemmatizer()

def get_min_parse(parsed_tree, leaf_index):
    """
    Get the minimum required component of a parsed tree for a certain leaf word.
    E.g. In the first step MAT ( MAT ) was hydrolyzed under constant stirring with a mixed solution of MAT and MAT and using MAT as catalyst ; molar ratio was 1 : 4 : 10 : 05.
    When inspecting the firt MAT word, the sentence would be converted to: 
    PP MAT PRN was hydrolyzed PP; S.

    :param parsed_tree: a parsed tree by Stanford Parser
    :param leaf_index: the index of the leaf word
    :return minParse: list of words or (sub-)tree objects forming a compact sentence about the leaf word 
    """    
    # goal
    minParse = []
    all_leaf_pos = parsed_tree.treepositions('leaves')
    pos_index_dict = {str(pos): i for (i, pos) in enumerate(all_leaf_pos)}
    # left side
    # tranverse from bottom to up, and expanding VP node if still in local sentence (no S label met)
    leaf_pos = parsed_tree.leaf_treeposition(leaf_index)
    position = list(leaf_pos)
    minParse.append((parsed_tree[position], pos_index_dict[str(leaf_pos)])) 
    last_position = position.pop()
    inSentence = True
    while len(position) > 0:
        if len(parsed_tree[position]) == 1 and last_position == 0:
            if parsed_tree[position].label() == 'S':
                inSentence = False
            last_position = position.pop()
        else:
            # assert len(parsed_tree[position]) > 1
            positionChanged = False
            for i in reversed(range(0, last_position)):
                node = parsed_tree[position][i]
                if node.height() == 2:
                    minParse.insert(0, (node[0], pos_index_dict[str(tuple(position+[i, 0]))]))
                elif node.label() == 'VP' and inSentence == True:
                    position.append(i)
                    last_position = len(node)
                    positionChanged = True
                    break
                else:
                    minParse.insert(0, node)
            if positionChanged == False:
                if parsed_tree[position] == 'S':
                    inSentence = False
                last_position = position.pop()

    # right side
    position = list(leaf_pos)
    last_position = position.pop()
    inSentence = True
    while len(position) > 0:
        if len(parsed_tree[position]) == 1 and last_position == 0:
            if parsed_tree[position].label() == 'S':
                inSentence = False
            last_position = position.pop()
        else:
            # assert len(parsed_tree[position]) > 1
            positionChanged = False
            for i in range(last_position+1, len(parsed_tree[position])):
                node = parsed_tree[position][i]
                if node.height() == 2:
                    minParse.append((node[0], pos_index_dict[str(tuple(position+[i, 0]))]))
                elif node.label() == 'VP' and inSentence == True:
                    position.append(i)
                    last_position = -1
                    positionChanged = True
                    break
                else:
                    minParse.append(node)
            if positionChanged == False:
                if parsed_tree[position] == 'S':
                    inSentence = False
                last_position = position.pop()

    return minParse

def find_phrase(dep_tree, start_address, end_address, parsed_tree, min_words = None, from_end = True):
    """
    Find the key words/phrases corresponding to the start node that has a direct or indirect dependency relation with the end note.
    It is VB or association of VB and IN here.

    :param dep_tree: a dependency tree from Stanford Dependency Parser
    :param start_address: address of start node, the node corresponding to the key word (VB)
    :param end_address: address of end node, the node corresponding to MAT
    :param parsed_tree: a parsed tree from Stanford Parser
    :param min_words: words in the compact sentence got from get_min_parse() 
    :param from_end: the direction of searching IN (from start node to end node or from end node to start node)
    :return phrase: a string which is a word or a phrase
    """    

    # goal
    phrase = ''
    # root
    node_start = dep_tree.get_by_address(start_address)
    # node corresponding to MAT
    node_end = dep_tree.get_by_address(end_address)

    # end_address is in the right side
    if end_address > start_address:
        # print('here')
        heads = [node_end]
        node_visiting = node_end
        while node_visiting['head'] != start_address:
            heads.insert(0, dep_tree.get_by_address(node_visiting['head']))
            node_visiting = heads[0]
        heads.insert(0, node_start)
        # position of start_address in parsed_tree
        position_start = parsed_tree.leaf_treeposition(start_address-1)
        words_IN = []
        address_endVB = -1
        for head in reversed(heads):
            if head['tag'].startswith('VB') or 1:
                # record address of end vb, in for other vb should before this address
                if address_endVB < 0 and head['tag'].startswith('VB'):
                    address_endVB = head['address']
                depNodes = []
                for dep_node in head['deps']:
                    depNodes.extend(head['deps'][dep_node])
                for depNode in depNodes:
                    # not visit right children
                    if depNode >= head['address']:
                        continue
                    # in is not before vb
                    if depNode <= start_address:
                        continue
                    # vb and in should in min_words sent
                    if min_words != None and (depNode-1 not in min_words):
                        continue
                    # vb and in should under the same vp node
                    position_dep = parsed_tree.leaf_treeposition(depNode-1)
                    if position_start[0:len(position_start)-2] != position_dep[0:len(position_start)-2]:
                        continue
                    node_visiting = dep_tree.get_by_address(depNode)
                    if node_visiting['tag'].startswith('IN'):
                        if (node_visiting['word'] in tabu_IN_words) and parsed_tree[position_dep[0:len(position_dep)-3]].label() == 'NP':
                            continue 
                        words_IN.append(node_visiting)
        if from_end == True:
            phrase = [node_start]+words_IN[0:1]
        elif len(words_IN) > 0 and words_IN[-1]['address'] < address_endVB:
            phrase = [node_start] + words_IN[-1:]
        else:
            phrase = [node_start]
        # phrase = [node_start]+words_IN        
        phrase = sorted(phrase, key=lambda node: node['address'])
        phrase = ' '.join([node['word'] for node in phrase])

    # end_address is in the left side
    else:
        phrase = node_start['word']
        pass
    return phrase

def word_regulate(word):
    """
    regulate a word to be able to be dealed with by stanford parser 
    
    :param word: word
    :return word_normal: regulated word with special characters replaced
    """
    word_normal = unidecode(word) 
    substitutionTable = {
        ord('.'): None,
        ord('%'): None,
        # ord('-'): None,
        ord('/'): None,
    }
    word_normal = word_normal.translate(substitutionTable)
    word_normal = word_normal.strip()
    return word_normal

def sent_regulate(sentence):
    """
    regulate words in a sentence to be able to be dealed with by stanford parser 
    
    :param sentence: list of words
    :param sent_normal: list of regulated words
    """
    sent_normal = []
    i = 0
    while (i < len(sentence)):
        word = sentence[i]
        word_normal = word_regulate(word)
        if len(word_normal) != 0:
            if (word_normal != '-'):
                sent_normal.append(word_normal)
            elif sent_normal[-1] != 'MAT' :
                j = i
                word2 = word_normal
                while word2 == '-':
                    sent_normal[-1] += word2  
                    j += 1
                    word2 = word_regulate(sentence[j])
                if word2 != 'MAT':  
                    sent_normal[-1] += word2
                else:
                    sent_normal[-1] = sent_normal[-1].strip('-')
                    sent_normal.append(word2)
                i = j 
        i += 1
    return sent_normal

def get_key_words(str_words):
    """
    Get the key words that have a direct or indirect dependency relation with a material word.
    
    :param str_words: list of words, which froms a sentence
    :return key_words: list of key words, which are usually VB or VB + IN in the sentence
    """
    # goal
    key_words = []
    for i in range(len(str_words)):
        key_words.append([])

    # reshape the words and add index
    sentence = []
    mat = []
    for (tmp_index, word) in enumerate(str_words):
        if word == '<MAT>':
            mat.append([len(sentence), tmp_index])
            sentence.append('MAT')
        else:
            sentence.append(word)

    sentence = sent_regulate(sentence)
    mat_index = -1
    for i in range(len(sentence)):
        if sentence[i] == 'MAT':
            mat_index += 1
            mat[mat_index][0] = i


    # parse data
    sent = ' '.join(sentence)
    # print(sent)
    # print(mats[i])

    parsed_sent = tree_parser.raw_parse(sent)
    parsed_sent = next(parsed_sent)
    # print(parsed_sent)
    # parsed_sent.draw()    

    # need to check a sent is not split into two sents
    try:
        parsed_dep = next(dep_parser.raw_parse(sent))
    except Exception as e:
        parsed_dep = None
        print(sent)

    mat_index = -1
    words = parsed_sent.leaves()

    for j, t in enumerate(words):
        if t != 'MAT':
            continue
        mat_index += 1
        if parsed_dep != None:            
            assert parsed_dep.contains_address(j+1)
            node = parsed_dep.get_by_address(j+1)
            minParse =  get_min_parse(parsed_sent, j) 
            minWords =  [w[1] for w in minParse if not isinstance(w, Tree)]
            head = node['head']
            while(node['head'] != 0):
                node = parsed_dep.get_by_address(node['head'])
                if node['tag'].startswith('VB'):
                    if node['address']-1 in minWords:
                        # use vb + in for all dep word
                        if len(key_words[mat[mat_index][1]]) == 0:
                            phrase = find_phrase(parsed_dep, node['address'], j+1, parsed_sent, minWords, from_end = True)
                        else:
                            phrase = find_phrase(parsed_dep, node['address'], j+1, parsed_sent, minWords, from_end = False)
                        if node['address'] > j+1:
                            key_words[mat[mat_index][1]].append('r_'+phrase)
                        else:
                            key_words[mat[mat_index][1]].append('l_'+phrase)
    return key_words