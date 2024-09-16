from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model (ensure this matches how you've saved your model)
models = joblib.load('well_log_v1.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the request
        data = request.get_json()
        print(f"Received data: {data}")

        # Convert input data to the correct format
        inputs = np.array(data)
        
        # Ensure the inputs have the correct shape and feature names
        if inputs.shape[1] != 2:  # Adjust based on the number of features
            raise ValueError("Invalid input shape")

        predictions = []
        
        # Process each sample in the input
        for sample in inputs:
            pseudo_tvd = sample.get('Pseudo_TVD')
            gr = sample.get('GR')

            if pseudo_tvd is None or gr is None:
                raise ValueError("Missing required features")

            # Predict NPHI
            rf_reg1 = models[0][0]
            nphi = rf_reg1.predict([[pseudo_tvd, gr]])[0]
            
            # Predict log_RD
            rf_reg2 = models[0][1]
            log_rd = rf_reg2.predict([[pseudo_tvd, gr, nphi]])[0]
            
            # Predict RHOB
            rf_reg3 = models[0][2]
            rhob = rf_reg3.predict([[pseudo_tvd, gr, nphi, log_rd]])[0]

            predictions.append([nphi, log_rd, rhob])

        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
