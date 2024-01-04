from flask import Flask, render_template, request
import pickle
import numpy as np

# Cream el diccionari de noms de Flors
Noms_flors = {
    0: 'Iris Setosa',
    1: 'Iris Versicolor',
    2: 'Iris Virginica'
}

# Cream la ruta a les imatges de les flors. 

Images_flors = {
    'Iris Setosa': 'images/0.jpg',
    'Iris Versicolor': 'images/1.jpg',
    'Iris Virginica': 'images/2.jpg'
}

app = Flask(__name__)

# Carregam els models serialitzats 
def load_model(model_name):
    with open(f'{model_name}_model.pkl', 'rb') as file:
        return pickle.load(file)

models = {
    "Regresio": load_model('Regresio'),
    "Svm": load_model('Svm'),
    "Arbre": load_model('Arbre'),
    "Knn": load_model('Knn')
}

# carregam el scaler 
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    input_data = np.array([[petal_length, petal_width]])

    # Escalam les dades de entrada
    input_data_scaled = scaler.transform(input_data)

    # feim la prediccio
    predictions = {model: models[model].predict(input_data_scaled)[0] for model in models}

    # Posam nom a les prediccions 
    predictions_named = {model: Noms_flors[prediction] for model, prediction in predictions.items()}

    # posam imatge a les flors
    predictions_images = {model: Images_flors[predictions_named[model]] for model in predictions_named}

    return render_template('results.html', predictions=predictions_named, images=predictions_images)

if __name__ == '__main__':
    app.run(debug=True)

