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
    long_petal= float(request.form['long_petal'])
    ample_petal = float(request.form['ample_petal'])
    input_data = np.array([[long_petal, ample_petal]])

    # Escalam les dades de entrada
    entrada_escalada = scaler.transform(input_data)

    # feim la prediccio
    predictions = {model: models[model].predict(entrada_escalada)[0] for model in models}

    # Posam nom a les prediccions 
    prediccio_amb_nom = {model: Noms_flors[prediction] for model, prediction in predictions.items()}

    # posam imatge a les flors
    prediccions_imagtes = {model: Images_flors[prediccio_amb_nom[model]] for model in prediccio_amb_nom}

    return render_template('results.html', predictions=prediccio_amb_nom, images=prediccions_imagtes)

if __name__ == '__main__':
    app.run(debug=True)

