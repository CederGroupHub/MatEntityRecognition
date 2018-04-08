import collections
import theano.tensor as T

# convert a dict into orderedDict
def dictOrdered(unordered_dict):
	return collections.OrderedDict(sorted(unordered_dict.items(), key=lambda x:x[0]))

# get all variables from theano tensor variable
def getAllVar(tensorVar):
	vars = set()
	if isinstance(tensorVar, T.TensorVariable):
		if tensorVar.owner == None:
			vars.add(tensorVar)
		else:
			for tmp_tensorVar in tensorVar.owner.inputs:
				vars.update(getAllVar(tmp_tensorVar))
	return vars

# get all combination of possible parameters
def getAllParaCombo(paraMatrix):
    paraCombo = []
    firstKey = list(paraMatrix.keys())
    firstKey = firstKey[0]
    if len(paraMatrix) == 1:
        for tmp_para in paraMatrix[firstKey]:
            new_combo = {}
            new_combo[firstKey] = tmp_para
            paraCombo.append(new_combo)
    if len(paraMatrix) > 1:
        paraMatrixCopy = paraMatrix.copy()
        paraMatrixCopy.pop(firstKey)
        combos = getAllParaCombo(paraMatrixCopy) 
        for tmp_para in paraMatrix[firstKey]:
            for tmp_combo in combos:
                new_combo = tmp_combo.copy()
                new_combo[firstKey] = tmp_para
                paraCombo.append(new_combo)
    return paraCombo