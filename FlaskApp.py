#Imports
from flask import Flask, json, jsonify, render_template, request
import numpy as np

app = Flask(__name__)

@app.route('/')
def homePage():
    return app.send_static_file('Draw.html')

if __name__ == "__main__":
    app.run(debug = True, threaded = False)