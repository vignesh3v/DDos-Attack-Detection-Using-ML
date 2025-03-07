import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pickle
import warnings
from flask import Flask, render_template, request

warnings.filterwarnings("ignore")

data = pd.read_csv("APA-DDOS-Dataset.csv")
print(data)

le = LabelEncoder()
data['frame.time'] = le.fit_transform(data['frame.time'])
data['ip.dst'] = le.fit_transform(data['ip.dst'])
data['ip.src'] = le.fit_transform(data['ip.src'])
data['Label'] = le.fit_transform(data['Label'])

X = data.drop(['Label', 'frame.time'], axis=1)
Y = data['Label']

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Describe", X.describe())

x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.25, random_state=0)

NB = GaussianNB()
NB.fit(x_train, y_train)  # Train the model
y_pred = NB.predict(x_test)

print("Naive Bayes ACCURACY is", accuracy_score(y_test, y_pred))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("browser2.html")

@app.route('/login', methods=['POST'])
def login():
    uname = request.form['files']
    rr = pd.read_csv(uname)
    rr['ip.dst'] = le.fit_transform(rr['ip.dst'])
    rr['ip.src'] = le.fit_transform(rr['ip.src'])
    
    y_pre = NB.predict(rr)
    
    if y_pre[0] == 0:
        return render_template('index1.html')
    elif y_pre[0] == 1:
        return render_template('index2.html')
    elif y_pre[0] == 2:
        return render_template('index3.html')

if __name__ == "__main__":
    app.run(debug=True)
