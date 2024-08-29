import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load the trained model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function for making predictions
def predict_insurance_claim(input_data):
    prediction = model.predict(input_data)
    return prediction

# Define a function to load the label encoder and transform the input
def transform_diagnosis_code(input_code, encoder_file):
    loaded_label_encoder = joblib.load(encoder_file)
    if input_code:
        try:
            return loaded_label_encoder.transform([input_code])[0]
        except ValueError:
            return 0
    return 0

# Streamlit app
st.title('Insurance Claim Prediction')

# Input fields
st.header('Enter the following details:')

# Feature Values
admit_days = st.number_input('No. of days admitted in hospital (0 if outpatient)', min_value=0)
age = st.number_input('Age of Beneficiary', min_value=0)
ChronicCond_Alzheimer = 1 if st.selectbox('Do you have Alzheimer?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_Cancer = 1 if st.selectbox('Do you have Cancer?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_Depression = 1 if st.selectbox('Do you have Depression?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_Diabetes = 1 if st.selectbox('Do you have Diabetes?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_Heartfailure = 1 if st.selectbox('Do you have Heart Failure?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_IschemicHeart = 1 if st.selectbox('Do you have Ischemic Heart Disease?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_KidneyDisease = 1 if st.selectbox('Do you have Kidney Disease?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_ObstrPulmonary = 1 if st.selectbox('Do you have Obstructive Pulmonary Disease?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_Osteoporasis = 1 if st.selectbox('Do you have Osteoporosis?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_rheumatoidarthritis = 1 if st.selectbox('Do you have Rheumatoid Arthritis?', ['Yes', 'No']) == 'Yes' else 2
ChronicCond_stroke = 1 if st.selectbox('Have you had a Stroke?', ['Yes', 'No']) == 'Yes' else 2
claim_period_days = st.number_input('No. of days for which claim is reimbursed', min_value=0)

clm_admit_diagnosis_code = st.text_input('Claim Admission Diagnosis Code')
clm_diagnosis_code_1 = st.text_input('Claim Diagnosis Code 1')
clm_diagnosis_code_2 = st.text_input('Claim Diagnosis Code 2')
clm_diagnosis_code_3 = st.text_input('Claim Diagnosis Code 3')
clm_diagnosis_code_4 = st.text_input('Claim Diagnosis Code 4')
clm_diagnosis_code_5 = st.text_input('Claim Diagnosis Code 5')
clm_diagnosis_code_6 = st.text_input('Claim Diagnosis Code 6')
clm_diagnosis_code_7 = st.text_input('Claim Diagnosis Code 7')
clm_diagnosis_code_8 = st.text_input('Claim Diagnosis Code 8')
clm_diagnosis_code_9 = st.text_input('Claim Diagnosis Code 9')
clm_diagnosis_code_10 = st.text_input('Claim Diagnosis Code 10')

clm_admit_diagnosis_code = transform_diagnosis_code(clm_admit_diagnosis_code, 'label_encoders/label_encoder_ClmAdmitDiagnosisCode.pkl')
clm_diagnosis_code_1 = transform_diagnosis_code(clm_diagnosis_code_1, 'label_encoders/label_encoder_ClmDiagnosisCode_1.pkl')
clm_diagnosis_code_2 = transform_diagnosis_code(clm_diagnosis_code_2, 'label_encoders/label_encoder_ClmDiagnosisCode_2.pkl')
clm_diagnosis_code_3 = transform_diagnosis_code(clm_diagnosis_code_3, 'label_encoders/label_encoder_ClmDiagnosisCode_3.pkl')
clm_diagnosis_code_4 = transform_diagnosis_code(clm_diagnosis_code_4, 'label_encoders/label_encoder_ClmDiagnosisCode_4.pkl')
clm_diagnosis_code_5 = transform_diagnosis_code(clm_diagnosis_code_5, 'label_encoders/label_encoder_ClmDiagnosisCode_5.pkl')
clm_diagnosis_code_6 = transform_diagnosis_code(clm_diagnosis_code_6, 'label_encoders/label_encoder_ClmDiagnosisCode_6.pkl')
clm_diagnosis_code_7 = transform_diagnosis_code(clm_diagnosis_code_7, 'label_encoders/label_encoder_ClmDiagnosisCode_7.pkl')
clm_diagnosis_code_8 = transform_diagnosis_code(clm_diagnosis_code_8, 'label_encoders/label_encoder_ClmDiagnosisCode_8.pkl')
clm_diagnosis_code_9 = transform_diagnosis_code(clm_diagnosis_code_9, 'label_encoders/label_encoder_ClmDiagnosisCode_9.pkl')
clm_diagnosis_code_10 = transform_diagnosis_code(clm_diagnosis_code_10, 'label_encoders/label_encoder_ClmDiagnosisCode_10.pkl')

county = st.selectbox('Country', [0, 1, 10, 11, 14, 20, 25, 30, 34, 40, 50, 55, 60, 70, 80, 84, 88, 90, 100, 110, 111, 113, 117, 120, 130, 131, 140, 141, 150, 160, 161, 170, 180, 190, 191, 194, 200, 210, 211, 212, 213, 220, 221, 222, 223, 224, 230, 240, 241, 250, 251, 260, 270, 271, 280, 281, 288, 290, 291, 292, 300, 301, 310, 311, 312, 320, 321, 328, 330, 331, 340, 341, 342, 343, 350, 360, 361, 362, 370, 380, 381, 390, 391, 392, 400, 410, 411, 412, 420, 421, 430, 431, 440, 441, 450, 451, 460, 461, 462, 470, 471, 480, 490, 500, 510, 511, 520, 521, 522, 530, 531, 540, 541, 542, 550, 551, 552, 560, 561, 562, 563, 564, 570, 580, 581, 582, 583, 590, 591, 592, 600, 601, 610, 611, 612, 620, 621, 622, 630, 631, 632, 640, 641, 650, 651, 652, 653, 654, 660, 661, 662, 670, 671, 672, 680, 681, 690, 691, 700, 701, 702, 703, 710, 711, 712, 720, 722, 730, 731, 734, 740, 741, 742, 743, 744, 750, 751, 752, 753, 754, 755, 756, 757, 758, 760, 761, 770, 771, 772, 780, 782, 783, 784, 785, 790, 791, 792, 793, 794, 795, 796, 797, 800, 801, 802, 803, 804, 810, 811, 812, 820, 821, 822, 830, 831, 832, 834, 835, 838, 840, 841, 842, 843, 844, 845, 850, 851, 860, 861, 862, 867, 870, 871, 873, 874, 875, 876, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 890, 891, 892, 893, 900, 901, 902, 903, 904, 905, 910, 911, 912, 913, 920, 921, 930, 931, 932, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 960, 961, 962, 963, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 996, 999])
deductible_amt_paid = st.number_input('Deductible Amount Paid', min_value=0)
gender = 1 if st.selectbox('Gender', ['Male', 'Female']) == 'Male' else 2
ip_annual_deductible_amt = st.number_input('Inpatient Annual Deductible Amount', min_value=0)
ip_annual_reimbursement_amt = st.number_input('Inpatient Annual Reimbursement Amount', min_value=0)
no_of_months_part_a_cov = st.number_input('Number of Months Part A Covered (0-12)', min_value=0, max_value=12)
no_of_months_part_b_cov = st.number_input('Number of Months Part B Covered (0-12)', min_value=0, max_value=12)
op_annual_deductible_amt = st.number_input('Outpatient Annual Deductible Amount', min_value=0)
op_annual_reimbursement_amt = st.number_input('Outpatient Annual Reimbursement Amount', min_value=0)
provider = st.text_input('Provider')
provider = transform_diagnosis_code(provider, 'label_encoders/label_encoder_Provider.pkl')
race = st.selectbox('Race', [1,2,3,5])
renal_disease_indicator = 1 if st.selectbox('Renal Disease Indicator', ['Yes', 'No']) == 'Yes' else 0
state = st.number_input('State', min_value=1,max_value=54)


# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'AdmissionDischargePeriodDays': [admit_days],
    'Age': [age],
    'ChronicCond_Alzheimer': [ChronicCond_Alzheimer],
    'ChronicCond_Cancer': [ChronicCond_Cancer],
    'ChronicCond_Depression': [ChronicCond_Depression],
    'ChronicCond_Diabetes': [ChronicCond_Diabetes],
    'ChronicCond_Heartfailure': [ChronicCond_Heartfailure],
    'ChronicCond_IschemicHeart': [ChronicCond_IschemicHeart],
    'ChronicCond_KidneyDisease': [ChronicCond_KidneyDisease],
    'ChronicCond_ObstrPulmonary': [ChronicCond_ObstrPulmonary],
    'ChronicCond_Osteoporasis': [ChronicCond_Osteoporasis],
    'ChronicCond_rheumatoidarthritis': [ChronicCond_rheumatoidarthritis],
    'ChronicCond_stroke': [ChronicCond_stroke],
    'ClaimPeriodDays': [claim_period_days],
    'ClmAdmitDiagnosisCode': [clm_admit_diagnosis_code],
    'ClmDiagnosisCode_1': [clm_diagnosis_code_1],
    'ClmDiagnosisCode_10': [clm_diagnosis_code_2],
    'ClmDiagnosisCode_2': [clm_diagnosis_code_3],
    'ClmDiagnosisCode_3': [clm_diagnosis_code_4],
    'ClmDiagnosisCode_4': [clm_diagnosis_code_5],
    'ClmDiagnosisCode_5': [clm_diagnosis_code_6],
    'ClmDiagnosisCode_6': [clm_diagnosis_code_7],
    'ClmDiagnosisCode_7': [clm_diagnosis_code_8],
    'ClmDiagnosisCode_8': [clm_diagnosis_code_9],
    'ClmDiagnosisCode_9': [clm_diagnosis_code_10],
    'County': [county],
    'DeductibleAmtPaid': [deductible_amt_paid],
    'Gender': [gender],
    'IPAnnualDeductibleAmt': [ip_annual_deductible_amt],
    'IPAnnualReimbursementAmt': [ip_annual_reimbursement_amt],
    'NoOfMonths_PartACov': [no_of_months_part_a_cov],
    'NoOfMonths_PartBCov': [no_of_months_part_b_cov],
    'OPAnnualDeductibleAmt': [op_annual_deductible_amt],
    'OPAnnualReimbursementAmt': [op_annual_reimbursement_amt],
    'Provider': [provider],
    'Race': [race],
    'RenalDiseaseIndicator': [renal_disease_indicator],
    'State': [state]
})

# Make prediction when button is clicked
if st.button('Predict'):
    prediction = predict_insurance_claim(input_data)
    st.write(f'Predicted Insurance Claim Amount: {prediction[0]}')
