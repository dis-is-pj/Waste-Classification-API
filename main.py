
from flask import Flask
# from flask_ngrok import run_with_ngrok
from fastai.vision import load_learner, open_image
from flask_restful import Resource, Api
import pyrebase


config={
    "apiKey": "AIzaSyBumH3t0dgqUQfNRx0lhZdrp4UcA0s6r7o",
    "authDomain": "sih-db-3b091.firebaseapp.com",
    "databaseURL": "https://sih-db-3b091.firebaseio.com",
    "projectId": "sih-db-3b091",
    "storageBucket": "sih-db-3b091.appspot.com",
    "messagingSenderId": "826516287205",
    "appId": "1:826516287205:web:7421a9ecfb7ac8e7b02e0c"

}


learn = load_learner('')

def pred(filename):
    try:
        firebase=pyrebase.initialize_app(config)
        storage= firebase.storage()
        pathoncloud='images/'+filename
        pathlocal = '/tmp/image.jpg'
        storage.child(pathoncloud).download(pathlocal)
        img = open_image('/tmp/image.jpg')
        s = str(learn.predict(img)[0])
        return(s)
    except:
        return("error")

app = Flask(__name__)
api = Api(app)
#run_with_ngrok(app)

class GetPrediction(Resource):
    def get(self,filename):
        return(pred(filename))

api.add_resource(GetPrediction, '/<string:filename>')

if __name__ == '__main__':
    app.run()
