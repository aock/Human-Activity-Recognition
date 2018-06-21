import os
import json
from keras.models import model_from_json, load_model

def saveModel(model, modelname):
    if not os.path.exists(modelname):
        os.makedirs(modelname)
    model.save(modelname + '/entireModel.h5')

def readModel(modelname):
    return load_model(modelname + '/entireModel.h5')

def saveWeights(model, modelname):
    if not os.path.exists(modelname):
        os.makedirs(modelname)
    model.save_weights(modelname + '/weights.h5')

def readWeights(model, modelname):
    model.load_weights(modelname + '/weights.h5')

def saveArch(model, modelname):
    if not os.path.exists(modelname):
        os.makedirs(modelname)
    with open(modelname+'/arch.json', 'w') as f:
        f.write(model.to_json())

def readArch(modelname):
    with open(modelname + '/arch.json', 'r') as f:
        model = model_from_json(f.read())
    return model

def saveResults(modelname, results):
    if not os.path.exists(modelname):
        os.makedirs(modelname)
    with open(modelname + '/results.json', 'w') as f:
        json.dump(results, f)

def readResults(modelname):
    filename = modelname + '/results.json'
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return data
    else:
        return False