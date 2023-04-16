from flask import Flask,render_template,request
import pickle
import pandas as pd
model=pickle.load(open(r"C:\Users\ELCOT\Downloads\thyroid_1_model.pkl", 'rb'))
le=pickle.load(open(r"C:\Users\ELCOT\Downloads\label_encoder.pkl", 'rb'))

app = Flask(__name__)

@app.route("/")
def about():
    return render_template('home.html')
@app.route("/pred", methods=['POST'])
def predict():
    x = [[float(x) for x in request.form.values()]]

    print(x)
    col = ['goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    x = pd.DataFrame(x, columns=col)
    print(x)
    pred = model.predict()
    pred = le.inverse_transform(pred)
    print(pred[0])
    return render_template('Submit.html', prediction_text=str(pred))
if __name__ == "__main__":
    app.run(debug=False)
