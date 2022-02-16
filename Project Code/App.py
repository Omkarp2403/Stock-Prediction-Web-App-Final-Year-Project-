import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from datetime import date
from keras.models import load_model
import streamlit as st

start = "2015-01-01"
end = date.today().strftime ("%Y-%m-%d")


st.title("Stock Market Prediction")

user_input = st.text_input("Enter Stock Ticker","SBIN.NS")


df = data.DataReader(user_input,'yahoo',start,end)
#Describing Data
st.subheader("Date From 2015 to Today")
st.write(df.describe())
#Select Option
result=st.selectbox("Choose Price",['Select','Opening Price','Closing Price'])

#Visualization
if result == "Opening Price":
	st.subheader("Opening Price Vs Time chart with 100MA")
	Ma100 = df.Open.rolling(100).mean()
	fig = plt.figure(figsize =(12,6))
	plt.plot(df.Open)
	st.pyplot(fig)
	st.subheader("Opening Price Vs Time chart with 100MA & 200MA")
	Ma100 = df.Open.rolling(100).mean()
	Ma100 = df.Open.rolling(200).mean()
	fig = plt.figure(figsize =(12,6))
	plt.plot(df.Open)
	st.pyplot(fig)
	
	data_training = pd.DataFrame(df['Open'][0:int(len(df)*0.70)])
	data_testing = pd.DataFrame(df['Open'][int(len(df)*0.70): int(len(df))])

	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(feature_range=(0,1))
	data_training_array = scaler.fit_transform(data_training)
	
	model = load_model('keras_model.h5')
    # Testing Part
	past_100_days = data_training.tail(100)
	final_df = past_100_days.append(data_testing,ignore_index=True)
	input_data = scaler.fit_transform(final_df)

	x_test = []
	y_test = []

	for i in range(100,input_data.shape[0]):
	    x_test.append(input_data[i-100:i])
	    y_test.append(input_data[i,0])

	x_test,y_test = np.array(x_test),np.array(y_test)
	y_predicted = model.predict(x_test)
	scaler=scaler.scale_

	scale_factor = 1/scaler[0]
	y_predicted = y_predicted * scale_factor
	y_test = y_test * scale_factor

	#final graph

	st.subheader('Prediction vs Original')
	fig2 = plt.figure(figsize=(12,6))
	plt.plot(y_test,'b',label ="Original Price")
	plt.plot(y_predicted,"r",label = "Predicted Price")
	plt.xlabel("Time")
	plt.ylabel("Price")
	plt.legend()
	st.pyplot(fig2)

elif result == "Closing Price":
	st.subheader("Closing Price Vs Time chart with 100MA")
	ma100 = df.Close.rolling(100).mean()
	fig=plt.figure(figsize=(12,6))
	plt.plot(df.Close)
	st.pyplot(fig)
	st.subheader("Closing Price Vs Time chart with 100MA & 200MA")
	Ma100 = df.Close.rolling(100).mean()
	Ma100 = df.Close.rolling(200).mean()
	fig = plt.figure(figsize =(12,6))
	plt.plot(df.Close)
	st.pyplot(fig)

	data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
	data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(feature_range=(0,1))

	model = load_model('keras_model.h5')

    # Testing Part
	past_100_days = data_training.tail(100)
	final_df = past_100_days.append(data_testing,ignore_index=True)
	input_data = scaler.fit_transform(final_df)

	x_test = []
	y_test = []

	for i in range(100,input_data.shape[0]):
	    x_test.append(input_data[i-100:i])
	    y_test.append(input_data[i,0])

	x_test,y_test = np.array(x_test),np.array(y_test)

	y_predicted = model.predict(x_test)
	scaler=scaler.scale_

	scale_factor = 1/scaler[0]
	y_predicted = y_predicted * scale_factor
	y_test = y_test * scale_factor

	#final graph

	st.subheader('Prediction vs Original')
	fig2 = plt.figure(figsize=(12,6))
	plt.plot(y_test,'b',label ="Original Price")
	plt.plot(y_predicted,"r",label = "Predicted Price")
	plt.xlabel("Time")
	plt.ylabel("Price")
	plt.legend()
	st.pyplot(fig2)


else:
	print("No valid option selected")



