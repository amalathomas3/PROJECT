# -*- coding: utf-8 -*-


from flask import Flask, request, render_template
from flask_cors import cross_origin
from keras.models import load_model
import tensorflow as tf
import numpy as np
import pandas as pd
import dateutil.parser as dt
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
lstm_model = load_model('LSTM.h5')
# gru_model = load_model('GRU.h5')


@app.route("/")
@cross_origin()
def index():
    return render_template("index.html")

@app.route('/home.html', methods = ['GET'])
@cross_origin()
def home():
    return render_template("home.html")

def get_prediction(date_to_predict,model,look_back):
    path_to_data = "../dataset/BTC-Hourly.csv" 
    df = pd.read_csv(path_to_data,index_col='date', parse_dates=['date'])
    dataset = pd.DataFrame(df['close'])
    close_data=df['close'].values
    scaler = MinMaxScaler()
    close_data=scaler.fit_transform(dataset[['close']])
    close_data=close_data[::-1]
    dates=df.index
    close_data = close_data.reshape((-1))
    print(close_data)
    def predict(num_prediction, model):
        prediction_list = close_data[-look_back:]
        
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
            
        return prediction_list
        

    num_prediction = (dt.parse(date_to_predict)-dates[1]).days+1
    forecast = predict(num_prediction, model)
    forecast=scaler.inverse_transform(forecast.reshape(-1, 1))
    return forecast


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        print(request.form)
        # Date_to_predict
        date_to_predict = request.form["Date"].split('T')[0]
        # print('Inputs: ',gru_model.inputs)
        # gru_forecast,gru_forecast_dates=get_prediction(date_to_predict,gru_model,1)
        # print(gru_forecast,gru_forecast_dates)
        lstm_forecast=get_prediction(date_to_predict,lstm_model,10)
        print(lstm_forecast)
        pred_text=f"Bitcoin price is ${round(float(lstm_forecast[-1][0]), 2)} as per LSTM on {date_to_predict}"
        print(pred_text)
        return render_template('home.html',prediction_text=pred_text)


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)