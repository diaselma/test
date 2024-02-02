from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Charger les données
file_id = '10qeXNDBhc3ggpMe1ZsrnIhPXfU76Fj7m'
url = f'https://drive.google.com/uc?id={file_id}'
data = pd.read_csv(url)

# Séparer les caractéristiques (X) et les étiquettes (y)
X = data.drop('legitimate', axis=1)
y = data['legitimate']

# Charger le modèle entraîné
model_filename = 'C:\\Users\\ASUS\\Desktop\\MesModel\\randomForesModel.joblib'
loaded_model = joblib.load(model_filename)

# Noms des caractéristiques
feature_names = X.columns.tolist()

# Afficher la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Gérer la demande de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtenir les entrées de l'utilisateur depuis le formulaire
        user_input = []
        for feature in feature_names:
            user_input.append(float(request.form[feature]))

        # Convertir l'entrée en un tableau numpy
        input_array = np.array([user_input])

        # Faire une prédiction
        prediction = loaded_model.predict(input_array)

        # Afficher le résultat sur la page de résultat
        return render_template('index.html', prediction=prediction[0])

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(host="0.0.0.0", POST=5000)
