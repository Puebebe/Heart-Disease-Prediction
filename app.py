import streamlit as st
import pandas as pd
import pickle

model_pkl = pickle.load(open('model.pkl','rb'))
le_dict_pkl = pickle.load(open('le_dict.pkl', 'rb'))

def main():
    st.set_page_config(page_title='Heart Disease Prediction')
    st.header('Heart Disease Prediction')

    with st.form(key='my_form'):
        age = st.slider(label='Age', value=1, min_value=1, max_value=100)
        col1, col2, col3 = st.columns(3)
        sex = col1.selectbox(label='Sex', options=('M', 'F'))
        chest_pain_type = col2.selectbox(label='Chest-pain Type', options=('ASY', 'ATA', 'NAP', 'TA'))
        fasting_bs = col3.selectbox(label='Fasting Blood Sugar', options=(0, 1))
        
        col4, col5, col6 = st.columns(3)
        resting_ecg = col4.selectbox(label='RestingECG ', options=('LVH', 'Normal', 'ST'))
        exercise_angina = col5.selectbox(label='Exercise Angina', options=('N', 'Y'))
        st_slope = col6.selectbox(label='ST Slope', options=('Down', 'Flat', 'Up'))
        
        col7, col8, col9 = st.columns(3)
        resting_bp = col7.number_input(label='Resting Blood Pressure', min_value=1, step=1)
        max_heart_rate = col8.number_input(label='Max Heart Rate', min_value=1, step=1)
        cholesterol = col9.number_input(label='Cholesterol', min_value=1, step=1)
        
        submit = st.form_submit_button('Submit')
    
        if submit:
            data = pd.DataFrame({'Age': age,
                                 'Sex': sex,
                                 'ChestPainType': chest_pain_type,
                                 'RestingBP': resting_bp,
                                 'Cholesterol': cholesterol,
                                 'FastingBS': fasting_bs,
                                 'RestingECG': resting_ecg,
                                 'MaxHR': max_heart_rate,
                                 'ExerciseAngina': exercise_angina,
                                 'ST_Slope': st_slope},index=[0])
            
            for col in le_dict_pkl.keys():
                data[col] = le_dict_pkl[col].transform(data[col])
                
            prediction = model_pkl.predict(data)[0]
            st.subheader('Is individual suffering from a heart disease: {0}'.format('Yes' if prediction == 1 else 'No'))
            
    st.caption('s18581, s18490, s19978')
        
if __name__ == '__main__':
    main()