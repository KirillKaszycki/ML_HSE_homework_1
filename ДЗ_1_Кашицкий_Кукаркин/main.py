import streamlit as st
import keras
import pandas as pd
import numpy as np


model = keras.models.load_model('h1_classifier.h5')


def predict(model, input_data):
    prediction = model.predict(input_data)
    max_index = np.argmax(prediction)
    return max_index


if __name__ == '__main__':
    st.title('H1 Classifier')

    st.sidebar.header('Введите данные:')
    Unnamed = st.sidebar.number_input('Unnamed', min_value=0)
    orderID = st.sidebar.number_input('orderID', min_value=0)
    articleID = st.sidebar.number_input('articleID', min_value=0)
    colorCode = st.sidebar.number_input('colorCode', min_value=0)
    sizeCode = st.sidebar.number_input('sizeCode', min_value=0)
    productGroup = st.sidebar.number_input('productGroup', min_value=0)
    quantity = st.sidebar.number_input('quantity', min_value=0)
    price = st.sidebar.number_input('price', min_value=0.0)
    rrp = st.sidebar.number_input('rrp', min_value=0.0)
    voucherID = st.sidebar.number_input('voucherID', min_value=0)
    voucherAmount = st.sidebar.number_input('voucherAmount', min_value=0.0)
    customerID = st.sidebar.number_input('customerID', min_value=0)
    deviceID = st.sidebar.number_input('deviceID', min_value=0)
    paymentMethod = st.sidebar.number_input('paymentMethod', min_value=0)

    input_data = pd.DataFrame({
        'Unnamed: 0': [Unnamed],
        'orderID': [orderID],
        'articleID': [articleID],
        'colorCode': [colorCode],
        'sizeCode': [sizeCode],
        'productGroup': [productGroup],
        'quantity': [quantity],
        'price': [price],
        'rrp': [rrp],
        'voucherID': [voucherID],
        'voucherAmount': [voucherAmount],
        'customerID': [customerID],
        'deviceID': [deviceID],
        'paymentMethod': [paymentMethod]
    })

    if st.sidebar.button('Предсказать'):
        prediction = predict(model, input_data)
        st.markdown(f"<h1 style='text-align: center; color: red;'>Наиболее вероятный класс: {prediction}</h1>",
                    unsafe_allow_html=True)
