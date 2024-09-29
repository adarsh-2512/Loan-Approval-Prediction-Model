from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data from POST request
        no_of_dep = int(request.form['dependents'])
        grad = int(request.form['education'])
        self_emp = int(request.form['self_employed'])
        Annual_Income = float(request.form['income'])
        Loan_Amount = float(request.form['loan_amount'])
        Loan_Dur = int(request.form['loan_term'])
        Cibil = int(request.form['cibil'])
        Assets = float(request.form['assets'])

        # Prepare the data for prediction
        pred_data = pd.DataFrame([[no_of_dep, grad, self_emp, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                                 columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
        pred_data_scaled = scaler.transform(pred_data)

        # Make the prediction
        prediction = model.predict(pred_data_scaled)

        # Store the result in session
        if prediction[0] == 1:
            session['result'] = 'approved'
            session['message'] = 'Loan is Approved'
        else:
            session['result'] = 'rejected'
            session['message'] = 'Loan is Rejected'

        # Redirect to the same page
        return redirect(url_for('home'))

    # For GET request or after redirect, clear the result from session
    message = session.pop('message', None)
    result = session.pop('result', None)
    
    return render_template('loan_prediction.html', result=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
