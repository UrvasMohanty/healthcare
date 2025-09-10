# main.py - Unified AI Disease Prediction (tabular + skin image)
# Dependencies:
# pip install pandas scikit-learn tensorflow pillow numpy joblib

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import csv
from datetime import datetime


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib




# TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model, load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ---------------------- Tabular loaders (with debug) ----------------------

def load_cancer_data():
    df = pd.read_csv('Cancer_Data.csv')
    print(f"[Cancer] Raw shape: {df.shape}")
    df.replace('?', pd.NA, inplace=True)
    df = df.dropna()
    print(f"[Cancer] After dropna: {df.shape}")
    X = df.drop(columns=['id', 'diagnosis'], errors='ignore')
    y = df['diagnosis'].astype(int)
    return X, y

def load_heart_disease_data():
    df = pd.read_csv('heart_disease_data.csv')
    print(f"[Heart] Raw shape: {df.shape}")
    df = df.dropna()
    print(f"[Heart] After dropna: {df.shape}")
    X = df[['age', 'cp', 'trestbps', 'chol', 'thalach']]
    y = df['target'].astype(int)
    return X, y

def load_diabetes_data():
    df = pd.read_csv('diabetes.csv')
    print(f"[Diabetes] Raw shape: {df.shape}")
    df = df.dropna()
    print(f"[Diabetes] After dropna: {df.shape}")
    X = df[['Age','Insulin', 'Glucose', 'BloodPressure','DiabetesPedigreeFunction', 'BMI', ]]
    y = df['Outcome'].astype(int)
    return X, y

def load_kidney_data():
    df = pd.read_csv('kidney_disease.csv')
    print(f"[Kidney] Raw shape: {df.shape}")
    df.replace('?', pd.NA, inplace=True)
    df = df.dropna()
    print(f"[Kidney] After dropna: {df.shape}")
    X = df[['age', 'bp', 'bgr', 'su', 'al']].astype(float)
    y = (df['classification'] == 'ckd').astype(int)
    return X, y

def load_liver_data():
    df = pd.read_csv('indian_liver_patient.csv')
    print(f"[Liver] Raw shape: {df.shape}")
    df = df.dropna()
    print(f"[Liver] After dropna: {df.shape}")
    X = df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alamine_Aminotransferase', 'Alkaline_Phosphotase']]
    y = (df['Dataset'] == 2).astype(int)
    return X, y

def load_parkinsons_data():
    df = pd.read_csv('parkinsons.data')
    print(f"[Parkinson's] Raw shape: {df.shape}")
    df = df.dropna()
    print(f"[Parkinson's] After dropna: {df.shape}")
    X = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR']]
    y = df['status'].astype(int)
    return X, y

def load_cold_data():
    df = pd.read_csv('large_data.csv')
    print(f"[Cold] Raw shape: {df.shape}")
    df = df.dropna()
    print(f"[Cold] After dropna: {df.shape}")
    X = df[['sore_throat', 'runny_nose', 'sneezing', 'cough']]
    y = df['cold'].astype(int)
    return X, y

def load_fever_data():
    df = pd.read_csv('large_data.csv')
    print(f"[Fever] Raw shape: {df.shape}")
    df = df.dropna()
    print(f"[Fever] After dropna: {df.shape}")
    X = df[['temperature', 'chills', 'sweating', 'body_ache']]
    y = df['fever'].astype(int)
    return X, y

def load_flu_data():
    df = pd.read_csv('large_data.csv')
    print(f"[Flu] Raw shape: {df.shape}")
    df = df.dropna()
    print(f"[Flu] After dropna: {df.shape}")
    X = df[['fever', 'chills', 'fatigue', 'cough']]
    y = df['flu'].astype(int)
    return X, y

def load_malaria_data():
    df = pd.read_csv('Malaria_Dataset.csv')
    print(f"[Malaria] Raw shape: {df.shape}")
    df = df.dropna()
    print(f"[Malaria] After dropna: {df.shape}")
    X = df[['fever', 'headache', 'sweating', 'nausea']]
    y = df['malaria'].astype(int)
    return X, y

# ---------------------- Utilities ----------------------

def train_and_evaluate_tabular(X, y):
    if X.empty or y.empty:
        raise ValueError("Dataset is empty after loading/cleaning.")
    if X.shape[0] < 2:
        raise ValueError("Not enough data to split. Need at least 2 rows.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

def get_user_input(fields):
    print("\nPlease enter the following values (numeric):")
    vals = []
    for f in fields:
        val = input(f"{f}: ").strip().strip('"').strip("'")
        try:
            v = float(val)
            vals.append(v)
        except ValueError:
            print("❌ Invalid input. Please enter numeric values only.")
            return None
    return pd.DataFrame([vals], columns=fields)

# ---------------------- Skin Image Model Helpers ----------------------

SKIN_MODEL_FILE = "skin_disease_model.h5"
SKIN_CLASSES_FILE = "skin_classes.json"

def build_skin_model(num_classes, input_shape=(128,128,3)):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    # freeze base
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_skin_model(dataset_dir, classes=None, target_size=(128,128), batch_size=16, epochs=6):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not installed. Install tensorflow to train skin model.")
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    # infer classes from subfolders if not provided
    if classes is None:
        classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if len(classes) == 0:
        raise ValueError("No class subfolders found in dataset directory.")
    num_classes = len(classes)
    print(f"[Skin Train] Using classes: {classes}")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                 rotation_range=20, horizontal_flip=True, zoom_range=0.2)

    train_gen = datagen.flow_from_directory(
        dataset_dir, target_size=target_size, batch_size=batch_size, classes=classes, subset='training'
    )
    val_gen = datagen.flow_from_directory(
        dataset_dir, target_size=target_size, batch_size=batch_size, classes=classes, subset='validation'
    )

    model = build_skin_model(num_classes=num_classes, input_shape=(target_size[0], target_size[1], 3))
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(SKIN_MODEL_FILE)

    # save classes mapping so we can reuse it when loading
    with open(SKIN_CLASSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print(f"[Skin Train] Saved model to {SKIN_MODEL_FILE} and classes to {SKIN_CLASSES_FILE}")
    return model, classes

def load_skin_model_and_classes():
    if not TF_AVAILABLE:
        return None, None
    if os.path.exists(SKIN_MODEL_FILE) and os.path.exists(SKIN_CLASSES_FILE):
        model = load_model(SKIN_MODEL_FILE)
        with open(SKIN_CLASSES_FILE, 'r', encoding='utf-8') as f:
            classes = json.load(f)
        return model, classes
    return None, None

def predict_skin_image(image_path, model, classes, target_size=(128,128), top_k=3):
    # normalize path (strip quotes)
    image_path = image_path.strip().strip('"').strip("'")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = load_img(image_path, target_size=target_size)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]  # shape (num_classes,)
    top_indices = preds.argsort()[-top_k:][::-1]
    results = [(classes[i], float(preds[i])) for i in top_indices]
    return results  # list of (label, prob)

def print_recommendations(disease_name):
    doctors = {
        "skin_disease": [
            "Dr. Pradeep Kumar Sahu - +91 94372 22898",
            "Dr. Prativa Dash - +91 674 230 0570",
            "Dr. Abhijit Panda - +91 674 238 2000",
            "Dr. Rashmi Pradhan - +91 82600 77222 ",
            "Dr. Anup Kumar Sahoo  - +91 674 666 0100"
        ]
    }
    hospitals = {
        "skin_disease": [
            "Kalinga Institute of Medical Sciences (KIMS) - 📞 +91 674 230 4400",
            "Apollo Hospitals Bhubaneswar - 📞 +91 674-666-1016",
            "SUM Hospital - 📞 +91-63729 94543",
            "Sparsh Hospital & Critical Care, Bhubaneswar - 📞 +91 94371 06412",
            "Care Hospital - 📞 +91 99371 19821"
        ]
    }

    print("\n🏥 Recommended Doctors:")
    for d in doctors.get(disease_name, []):
        print(f"- {d}")

    print("\n🏥 Recommended Hospitals:")
    for h in hospitals.get(disease_name, []):
        print(f"- {h}")

HISTORY_FILE = "prediction_history.csv"

def log_prediction(patient_name, patient_age, disease, result, confidence=None):
    """Save each prediction result to a CSV history file."""
    file_exists = os.path.isfile(HISTORY_FILE)

    with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header only if file does not exist or is empty
        if not file_exists or os.path.getsize(HISTORY_FILE) == 0:
            writer.writerow(["Timestamp", "Patient Name", "Age", "Disease", "Result", "Confidence/Accuracy"])

        writer.writerow([
            datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            patient_name,
            patient_age,
            disease,
            result,
            f"{confidence:.2f}" if confidence is not None else "N/A"
        ])


# def view_prediction_history():
#     """Display saved prediction history in the terminal."""
#     if not os.path.isfile(HISTORY_FILE):
#         print("📂 No prediction history found yet.")
#         return
    
#     df = pd.read_csv(HISTORY_FILE)
#     print("\n📜 Prediction History:")
#     print(df.to_string(index=False))


# ---------------------- Main Program ----------------------

def main():
    diseases = {
        'cancer': ("Cancer", load_cancer_data),
        'heart': ("Heart Disease", load_heart_disease_data),
        'diabetes': ("Diabetes", load_diabetes_data),
        'kidney': ("Kidney Disease", load_kidney_data),
        'liver': ("Liver Disease", load_liver_data),
        'parkinson': ("Parkinson's Disease", load_parkinsons_data),
        'cold': ("Common Cold", load_cold_data),
        'fever': ("Fever", load_fever_data),
        'flu': ("Flu", load_flu_data),
        'malaria': ("Malaria", load_malaria_data),
        'skin': ("Skin Disease (image)", None)  # handled separately
    }

    print("🔬 AI Disease Prediction System (Tabular + Skin Image)\n")
    print("Supported (tabular):", ', '.join([k for k in diseases.keys() if k != 'skin']))
    print("Also supports: skin (image-based detection)\n")

    # print("📌 Menu:")
    # print("1. Run a Disease Prediction")
    # print("2. View Prediction History")
    # print("3. Exit")

    # choice = input("👉 Enter your choice (1/2/3): ").strip()
    # if choice == "2":
    #     view_prediction_history()
    #     return
    # elif choice == "3":
    #     print("👋 Exiting program.")
    #     return
    # elif choice != "1":
    #     print("❌ Invalid choice! Exiting.")
    #     return

    prompt = input("🧠 Enter a prompt (e.g., 'Do I have diabetes?' or 'skin'): ").lower().strip()
    patient_name = input("👤 Enter your Name: ").strip()
    patient_age = input("🎂 Enter your Age: ").strip()

    # If user asked for skin
    if 'skin' in prompt or prompt == 'skin':
        if not TF_AVAILABLE:
            print("❌ TensorFlow not installed. Install tensorflow to enable skin image detection.")
            return

        # Try to load saved model + classes
        model, classes = load_skin_model_and_classes()
        if model is None:
            print("⚠️ No saved skin model found (skin_disease_model.h5).")
            choice = input("Do you want to (t)rain a new model now (requires skin_dataset/ folder) or (q)uit? [t/q]: ").lower().strip()
            if choice == 't':
                dataset_dir = input("Enter path to skin image dataset folder (e.g., './skin_dataset'): ").strip().strip('"').strip("'")
                try:
                    model, classes = train_skin_model(dataset_dir)
                except Exception as e:
                    print("❌ Training failed:", e)
                    return
            else:
                print("Exiting skin detection.")
                return

        # Predict on an image
        image_path = input("Enter path to skin image file to analyze: ").strip().strip('"').strip("'")
        try:
            results = predict_skin_image(image_path, model, classes, top_k=3)
            
            top_label = None
            top_prob = 0.0
            if results:
                first = results[0]

                if isinstance(first, dict) and 'label' in first and 'prob' in first:
                    top_label = first['label']
                    top_prob = first['prob']
                    results = [(r['label'], r['prob']) for r in results if 'label' in r and 'prob' in r]

                elif isinstance(first, (list, tuple)) and len(first) >= 2:
                    top_label = first[0]
                    top_prob = first[1]

                else:
                    # If prediction is just labels
                    top_label = str(first)
                    top_prob = 1.0  # Assume 100% if no probability given
                    results = [(top_label, top_prob)]
            print("\n🔎 Top predictions:")
            for label, prob in results:
                print(f" - {label} : {prob*100:.2f}%")
            
            print("\n📋 Diagnosis Summary:")
            if "healthy" in top_label.lower():
                print(f"🟢 This image most likely shows **healthy skin** ({top_prob*100:.2f}% confidence).")
                print("✅ No signs of disease detected, but if you have symptoms, consult a dermatologist for confirmation.")
                log_prediction(patient_name, patient_age, "Skin Disease", "Healthy", top_prob*100)
            else:
                print(f"🔴 This image most likely indicates **{top_label}** ({top_prob*100:.2f}% confidence).")
                print("⚠️ Please consult a dermatologist for further evaluation and treatment.")
                log_prediction(patient_name, patient_age, "Skin Disease", top_label, top_prob*100)


            user_choice1 = input("Would you like me to recommend the best doctors and hospitals available in Odisha? (yes/no): ").strip().lower()
            if user_choice1 in ["yes","y"]:
                print_recommendations('skin_disease')  # generic skin recs
            else:
                print("Got it! You don't want recommendation from us.")
        except Exception as e:
            print("❌ Error during prediction:", e)
        return

    # Otherwise handle tabular diseases
    for key, (name, loader) in diseases.items():
        if key in prompt and key != 'skin':
            print(f"\n🔄 Loading and training model for {name}...")
            try:
                X, y = loader()
                print(f"[INFO] Data loaded: {X.shape[0]} rows, {X.shape[1]} features.")
                model, acc = train_and_evaluate_tabular(X, y)
                print(f"✅ Model trained with accuracy: {acc*100:.2f}%")

                # optionally save tabular model for reuse
                # joblib.dump(model, f"{key}_dt_model.joblib")

                fields = X.columns.tolist()
                user_df = get_user_input(fields)
                if user_df is not None:
                    prediction = int(model.predict(user_df)[0])
                    if prediction == 0:
                        print(f"🟢 You are Fit!\nYou are not suffering from {name}.")
                        log_prediction(patient_name, patient_age, name, "Fit", acc*100)
                    else:
                        print(f"🔴 You are at high chance of suffering from {name}!\nYou should consult a doctor for better treatment.")
                        log_prediction(patient_name, patient_age, name, "Diseased", acc*100)

                    
                        # ---------------------- Recommendations ----------------------
                        user_choice = input("Would you like me to recommend the best doctors and hospitals available in Odisha? (yes/no): ").strip().lower()
                        if user_choice in ["yes","y"]:
                            top_doctors = {
                                    'cancer': [
                                        ("Dr. Manoj Kumar Sahu", "Oncologist", "AIIMS Bhubaneswar", "0674-2476789"),
                                        ("Dr. Smita Nayak", "Oncologist", "KIMS Hospital", "0674-7105555"),
                                        ("Dr. Subrat Mohanty", "Oncologist", "SUM Hospital", "0674-2300200"),
                                        ("Dr. Satyajit Mohanty", "Oncologist", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Ramesh Chandra Sahu", "Oncologist", "SCB Medical College", "0671-2414300"),
                                    ],
                                    'heart': [
                                        ("Dr. Anupam Jena", "Cardiologist", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Abhijit Sahoo", "Cardiologist", "KIMS", "0674-7105555"),
                                        ("Dr. Santosh Satpathy", "Cardiologist", "SUM Ultimate Medicare", "1800-120-0000"),
                                        ("Dr. Ranjan Panda", "Cardiologist", "AIIMS Bhubaneswar", "0674-2476789"),
                                        ("Dr. Ashok Kumar Das", "Cardiologist", "Care Hospitals", "0674-6699999"),
                                    ],
                                    'diabetes': [
                                        ("Dr. Ranjita Mohanty", "Endocrinologist", "KIMS", "0674-7105555"),
                                        ("Dr. Shakti Ranjan Satpathy", "Diabetologist", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Rajeev Nayak", "Endocrinologist", "AIIMS", "0674-2476789"),
                                        ("Dr. Sudhanshu Sekhar", "Diabetologist", "SUM Hospital", "0674-2300200"),
                                        ("Dr. Swagatika Samal", "Endocrinologist", "SCB Medical College", "0671-2414300"),
                                    ],
                                    'kidney': [
                                        ("Dr. Satya Ranjan Sahu", "Nephrologist", "SUM Hospital", "0674-2300200"),
                                        ("Dr. Lalatendu Das", "Nephrologist", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Amrit Kumar", "Nephrologist", "AIIMS Bhubaneswar", "0674-2476789"),
                                        ("Dr. Shantanu Panda", "Nephrologist", "KIMS", "0674-7105555"),
                                        ("Dr. Prabhat Ranjan", "Nephrologist", "SCB Medical College", "0671-2414300"),
                                    ],
                                    'liver': [
                                        ("Dr. Debasis Panigrahi", "Gastroenterologist", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Chandrasekhar Mishra", "Hepatologist", "KIMS", "0674-7105555"),
                                        ("Dr. Ritesh Mohapatra", "Liver Specialist", "SUM Hospital", "0674-2300200"),
                                        ("Dr. Ananya Mohanty", "Gastroenterologist", "AIIMS", "0674-2476789"),
                                        ("Dr. Sanjay Tripathy", "Hepatologist", "SCB Medical", "0671-2414300"),
                                    ],
                                    'parkinson': [
                                        ("Dr. Debasis Panda", "Neurologist", "SCB Medical College", "0671-2414300"),
                                        ("Dr. Ajay Behera", "Neurologist", "AIIMS Bhubaneswar", "0674-2476789"),
                                        ("Dr. S.K. Mishra", "Neurologist", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. A. Tripathy", "Neurologist", "KIMS", "0674-7105555"),
                                        ("Dr. Sweta Nayak", "Neurologist", "SUM Hospital", "0674-2300200"),
                                    ],
                                    'cold': [
                                        ("Dr. Jayanti Panda", "General Physician", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Sandeep Pattnaik", "ENT Specialist", "KIMS", "0674-7105555"),
                                        ("Dr. Priya Das", "General Physician", "AIIMS", "0674-2476789"),
                                        ("Dr. Soumya Pradhan", "ENT Specialist", "SUM Hospital", "0674-2300200"),
                                        ("Dr. Bijay Nayak", "General Physician", "SCB Medical College", "0671-2414300"),
                                    ],
                                    'fever': [
                                        ("Dr. Amit Raj", "General Physician", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Rakesh Tripathy", "General Physician", "KIMS", "0674-7105555"),
                                        ("Dr. Ipsita Behera", "Infectious Disease", "AIIMS", "0674-2476789"),
                                        ("Dr. Sarita Nayak", "General Physician", "SCB", "0671-2414300"),
                                        ("Dr. Tapas Sahu", "General Physician", "SUM Hospital", "0674-2300200"),
                                    ],
                                    'flu': [
                                        ("Dr. Pritam Sahu", "General Physician", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Rina Nayak", "Infectious Disease", "AIIMS", "0674-2476789"),
                                        ("Dr. Niraj Tripathy", "General Physician", "KIMS", "0674-7105555"),
                                        ("Dr. Arpita Behera", "General Physician", "SCB", "0671-2414300"),
                                        ("Dr. Jyoti Patra", "General Physician", "SUM Hospital", "0674-2300200"),
                                    ],
                                    'malaria': [
                                        ("Dr. P.K. Mohapatra", "Infectious Disease", "SCB Medical College", "0671-2414300"),
                                        ("Dr. Chittaranjan Nayak", "General Physician", "AIIMS Bhubaneswar", "0674-2476789"),
                                        ("Dr. Meena Sahu", "General Physician", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Gitanjali Das", "Infectious Disease", "SUM Hospital", "0674-2300200"),
                                        ("Dr. Saurav Jena", "General Physician", "KIMS", "0674-7105555"),
                                    ],
                                    'skin': [
                                        ("Dr. Bibhuti B. Mohanty", " Dermatologist", "Dermacare Skin & Hair Clinic, Bhubaneswar", "+91-93371 43446"),
                                        ("Dr. Pradeep Kumar Sahoo", "Dermatologist & Cosmetologist", " Sparsh Hospital, Bhubaneswar", "+91 94371 06412"),
                                        ("Dr. Debasish Rath", "Dermatologist", "Apollo Hospitals", "0674-6661010"),
                                        ("Dr. Smita Mohanty", "Skin Specialist", "Hi-Tech Medical College & Hospital, Bhubaneswar", "+91 98610 98050"),
                                        ("Dr. Satyajit Sahu", "Dermatologist & Trichologist", " SUM Ultimate Medicare, Bhubaneswar", "+91 1800 120 8989"),
                                    ]
                            }
                            top_hospitals = {
                                    'cancer': [
                                        ("AIIMS Bhubaneswar", "0674-2476789"),
                                        ("KIMS Hospital", "0674-7105555"),
                                        ("SUM Hospital", "0674-2300200"),
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("Acharya Harihar Regional Cancer Centre, Cuttack", "+91-671-2414985"),
                                    ],
                                    'heart': [
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("KIMS", "0674-7105555"),
                                        ("SUM Ultimate Medicare", "1800-120-0000"),
                                        ("AIIMS Bhubaneswar", "0674-2476789"),
                                        ("Care Hospitals", "0674-6699999"),
                                    ],
                                    'diabetes': [
                                        ("KIMS", "0674-7105555"),
                                        ("Hi-Tech Medical College, Bhubaneswar", "+91-674-2370591"),
                                        ("AIIMS Bhubaneswar", "0674-2476789"),
                                        ("SUM Hospital", "0674-2300200"),
                                        ("SCB Medical College", "0671-2414300"),
                                    ],
                                    'kidney': [
                                        ("SUM Hospital", "0674-2300200"),
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("AIIMS Bhubaneswar", "0674-2476789"),
                                        ("KIMS", "0674-7105555"),
                                        ("SCB Medical College", "0671-2414300"),
                                    ],
                                    'liver': [
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("KIMS", "0674-7105555"),
                                        ("SUM Hospital", "0674-2300200"),
                                        ("AIIMS Bhubaneswar", "0674-2476789"),
                                        ("SCB Medical College", "0671-2414300"),
                                    ],
                                    'parkinson': [
                                        ("SCB Medical College", "0671-2414300"),
                                        ("AIIMS Bhubaneswar", "0674-2476789"),
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("KIMS", "0674-7105555"),
                                        ("SUM Hospital", "0674-2300200"),
                                    ],
                                    'cold': [
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("KIMS", "0674-7105555"),
                                        ("AIIMS Bhubaneswar", "0674-2476789"),
                                        ("SUM Hospital", "0674-2300200"),
                                        ("SCB Medical College", "0671-2414300"),
                                    ],
                                    'fever': [
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("KIMS", "0674-7105555"),
                                        ("AIIMS Bhubaneswar", "0674-2476789"),
                                        ("SCB Medical College", "0671-2414300"),
                                        ("SUM Hospital", "0674-2300200"),
                                    ],
                                    'flu': [
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("Capital Hospital, Bhubaneswar", "+91-674-2391983"),
                                        ("KIMS", "0674-7105555"),
                                        ("SCB Medical College", "0671-2414300"),
                                        ("SUM Hospital", "0674-2300200"),
                                    ],
                                    'malaria': [
                                        ("SCB Medical College", "0671-2414300"),
                                        ("AIIMS Bhubaneswar", "0674-2476789"),
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("SUM Hospital", "0674-2300200"),
                                        ("KIMS", "0674-7105555"),
                                    ],
                                    'skin': [
                                        ("Sparsh Hospital & Critical Care, Bhubaneswar", "+91 94371 06412"),
                                        ("Care Hospitals, Bhubaneswar", "+91 99371 19821"),
                                        ("Apollo Hospitals", "0674-6661010"),
                                        ("SUM Hospital", "0674-2300200"),
                                        ("KIMS", "+91 674 230 4400"),
                                    ]
                            }


                                

                            if key in top_doctors:
                                print("\n👨‍⚕️ Top 5 Doctors in Odisha:")
                                for i, (doc_name, specialty, hospital, contact) in enumerate(top_doctors[key], start=1):
                                    print(f"{i}. {doc_name} ({specialty}) — {hospital} (📞 {contact})")

                            if key in top_hospitals:
                                print("\n🏥 Top 5 Hospitals in Odisha:")
                                for i, (hospital_name, contact) in enumerate(top_hospitals[key], start=1):
                                    print(f"{i}. {hospital_name} (📞 {contact})")
                        else:
                            print("Got it! You don't want recommendation from us.")
               
                    
            except FileNotFoundError:
                print("❌ CSV file not found. Please ensure the dataset exists.")
            except ValueError as ve:
                print("❌ Error:", ve)
            except Exception as e:
                print("❌ Unexpected error:", e)
            return

    print("❌ Disease not recognized in your input. Try again with a valid disease keyword.")
    print("ℹ️ Supported diseases:", ', '.join(diseases.keys()))

if __name__ == "__main__":
    main()
