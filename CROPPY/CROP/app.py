from flask import Flask, request, render_template
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

# Define normal ranges for the features
normal_ranges = {
    "Nitrogen": (0, 200),
    "Phosphorus": (0, 150),
    "Potassium": (0, 150),
    "Temperature": (0, 50),  # degrees Celsius
    "Humidity": (0, 100),    # percentage
    "pH": (3.5, 10.0),       # soil pH range
    "Rainfall": (0, 500)     # mm
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Collect input values
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Check if inputs are within normal ranges
        feature_values = {"Nitrogen": N, "Phosphorus": P, "Potassium": K,
                          "Temperature": temp, "Humidity": humidity, "pH": ph, "Rainfall": rainfall}

        for feature, value in feature_values.items():
            if not (normal_ranges[feature][0] <= value <= normal_ranges[feature][1]):
                result = f"With the value of {feature} being {value}, no crop can be grown."
                return render_template('index.html', result=result)

        # Prepare features for prediction
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)

        # Map prediction to crop
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

        return render_template('index.html', result=result)

    except ValueError:
        return render_template('index.html', result="Invalid input. Please enter numeric values for all fields.")


if __name__ == "__main__":
    app.run(debug=True)
