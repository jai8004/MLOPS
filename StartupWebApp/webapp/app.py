from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
profit_model = joblib.load("profit_prediction.pk")


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
   
    if request.method == 'POST':
        try:
                 
            RnD_Spend = int(request.form['RnD_Spend'])
            Admin_Spend = int(request.form['Admin_Spend'])
            Market_Spend = int(request.form['Market_Spend'])

            test_feature = [RnD_Spend, Admin_Spend, Market_Spend]
            test_feature_arr = np.array(test_feature)
            test_feature_arr = test_feature_arr.reshape(1, -1)
            profit_prediction = profit_model.predict(test_feature_arr )
            profit = str(round(float(profit_prediction[0]), 2))
         
        except ValueError:
           return "Please check if the values are entered correctly"

       
    return render_template('predict.html', prediction = profit)



if __name__ == "__main__":
    app.run(host='0.0.0.0')
