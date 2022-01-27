import streamlit as st
import pandas as pd
import pickle

model_pkl = pickle.load(open('model.pkl','rb'))
le_dict_pkl = pickle.load(open('le_dict.pkl', 'rb'))

def main():
    st.set_page_config(page_title='Heart Disease Prediction')

    with st.form(key='my_form'):
        age = st.slider(label='Age', value=1, min_value=1, max_value=100)
        sex = st.selectbox(label='Sex', options=['M', 'F'])
        chest_pain_type = st.selectbox(label='Chest-pain Type', options=['ASY' 'ATA' 'NAP' 'TA'])
        resting_bp = st.number_input(label='Resting Blood Pressure')
        cholesterol = st.number_input(label='Cholesterol')
        fasting_bs = st.selectbox(label='Fasting Blood Sugar', options=[0, 1])
        resting_ecg = st.selectbox(label='RestingECG ', options=['LVH' 'Normal' 'ST'])
        max_heart_rate = st.number_input(label='Max Heart Rate')
        exercise_angina = st.selectbox(label='Exercise Angina', options=['N' 'Y'])
        st_slope = st.selectbox(label='ST Slope', options=['Down' 'Flat' 'Up'])
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
            st.write('hello')
	
if __name__ == "__main__":
    main()