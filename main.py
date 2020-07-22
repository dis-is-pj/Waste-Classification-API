
from flask import Flask
import requests
import os
#from flask_ngrok import run_with_ngrok
from fastai.vision import load_learner, open_image
from flask_restful import Resource, Api

learn = load_learner('')

def pred(filename):
    url = 'https://binary-cdk.herokuapp.com/static/uploads/' + filename
    try:
        r = requests.get(url, allow_redirects=True)
        open('/tmp/image.jpg','wb').write(r.content)
        img = open_image('/tmp/image.jpg')
        s = str(learn.predict(img)[0])
        return(s)
    except:
        return('Nahi ho Paya')

app = Flask(__name__)
api = Api(app)
#run_with_ngrok(app)

class GetPrediction(Resource):
    def get(self,filename):
        return(pred(filename))

api.add_resource(GetPrediction, '/<string:filename>')

if __name__ == '__main__':
    app.run()
