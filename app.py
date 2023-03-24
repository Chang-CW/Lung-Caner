from flask import Flask, request, jsonify
import pickle
import numpy as np
# from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#socketio = SocketIO(app, cors_allowed_origins='*')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
    loaded_model = pickle.load(open("pkl/lung_cancer_XGBClassifier.pkl", "rb"))
    print(loaded_model)
    result = loaded_model.predict_proba(to_predict)
    return round(result[0][1]*100, 2)

@app.route('/api', methods = ['GET'])
def returnProb():
    # d = {}
    # inputchr = str(request.args['query'])
    # answer = str(ord(inputchr))
    # d['output'] = answer
    # return d
    d = {}
    X = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
         'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
         'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
    to_predict_list = []
    for x in X:
        to_predict_list.append(int(request.args[x]))
    # print(to_predict_list)
    # to_predict_list.append((float(request.args['age'])-29)/48) 
    # to_predict_list.append(int(request.args['sex']))
    # to_predict_list.append(int(request.args['cp']))
    # to_predict_list.append((float(request.args['trtbps'])-94)/106)
    # to_predict_list.append((float(request.args['chol'])-126)/438)
    # to_predict_list.append(int(request.args['fbs']))
    # to_predict_list.append(int(request.args['restecg']))
    # to_predict_list.append((float(request.args['thalachh'])-71)/131)
    # to_predict_list.append(int(request.args['exng']))
    # to_predict_list.append(float(request.args['oldpeak']))
    # to_predict_list.append(int(request.args['slp']))
    # to_predict_list.append(int(request.args['caa']))
    # to_predict_list.append(int(request.args['thall']))
    d['output'] = str(ValuePredictor(to_predict_list))
    # return str(to_predict_list)
    return d

if __name__ =="__main__":
    app.run(debug=True)

# api?age=30&sex=0&cp=1&trtbps=1&chol=1&fbs=1&restecg=1&thalachh=1&exng=1&oldpeak=1&slp=1&caa=1&thall=1
