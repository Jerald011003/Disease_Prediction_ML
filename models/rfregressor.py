import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

train_data = pd.read_csv('datasets/Training.csv')
test_data = pd.read_csv('datasets/Testing.csv')

train_data = train_data.drop(columns=['Unnamed: 133'], errors='ignore')
test_data = test_data.drop(columns=['Unnamed: 133'], errors='ignore')


le = LabelEncoder()
train_data['prognosis'] = le.fit_transform(train_data['prognosis'])
test_data['prognosis'] = le.transform(test_data['prognosis'])

X_train = train_data.iloc[:, :-1].values  
y_train = train_data['prognosis'].values 

X_test = test_data.iloc[:, :-1].values 
y_test = test_data['prognosis'].values   

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f' ')
print(f'Model Metrics: ')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f' ')

symptom_categories = {
    "Head": ["headache", "dizziness", "loss_of_balance", "loss_of_smell", "visual_disturbances"],
    "Chest": ["cough", "chest_pain", "breathlessness", "fast_heart_rate", "phlegm"],
    "Abdomen": ["stomach_pain", "abdominal_pain", "nausea", "diarrhoea", "vomiting"],
    "Skin": ["itching", "skin_rash", "red_spots_over_body", "bruising", "scarring"]
    # ...
}

def predict_prognosis(model, symptoms_input):
    prediction = model.predict([symptoms_input])
    prognosis = le.inverse_transform(prediction)
    return prognosis[0]

def get_symptoms_input():
    print("Select the body part where you are experiencing symptoms:")
    for idx, body_part in enumerate(symptom_categories.keys()):
        print(f"{idx + 1}. {body_part}")
    
    body_part_choice = int(input("Enter the number corresponding to the body part: ")) - 1
    selected_body_part = list(symptom_categories.keys())[body_part_choice]
    
    symptoms = symptom_categories[selected_body_part]
    user_symptoms = [0] * len(train_data.columns[:-1])  
    
    print(f"\nAnswer '1' for Yes and '0' for No for the following symptoms in the {selected_body_part} area:")
    for symptom in symptoms:
        while True:
            try:
                user_input = input(f"Do you have {symptom.replace('_', ' ')}? (1 for Yes, 0 for No): ")
                if user_input not in ['0', '1']:
                    raise ValueError("Please enter 1 or 0")
                symptom_index = train_data.columns.get_loc(symptom)
                user_symptoms[symptom_index] = int(user_input)
                break
            except ValueError as e:
                print(e)
    
    return user_symptoms

user_symptoms = get_symptoms_input()

predicted_disease = predict_prognosis(model, user_symptoms)
print(f'Predicted Disease: {predicted_disease}')
