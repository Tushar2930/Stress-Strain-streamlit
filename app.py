import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model


def app():
    df = pd.read_csv('final_explo.csv')
    exp_strain = np.array(df['exp_strain'])
    exp_stress = np.array(df['exp_stress'])
    scalar = pickle.load(open('scaling.pkl', 'rb'))
    model = load_model()
    st.title("Stress Strain Error Estimation")
    # take 3 input values from user
    st.write("Enter the values of A, B and n")
    A = st.number_input("A", min_value=0, max_value=515)
    B = st.number_input("B", min_value=0, max_value=3500)
    n = st.number_input("n", min_value=0.00, max_value=3.00)
    scaled_input = scalar.transform([[A, B, n]])
    calc_stress = A+B*(np.power(exp_strain, n))
    plt.plot(exp_strain, exp_stress, 'r', label='Experimental')
    plt.plot(exp_strain, calc_stress, 'b', label='Calculated')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.legend()
    st.write("The graph of the stress strain curve is shown below")
    st.pyplot(plt)
    st.write("Actual MAE : ", mae(exp_stress, calc_stress))
    st.write("Predicted MAE by ANN: ", model.predict(scaled_input)[0][0])


if __name__ == '__main__':
    app()
