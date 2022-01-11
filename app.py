from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from csv import DictWriter
import datetime
app = Flask(__name__)


def csv_writer(data):
    headerscsv = ['Name', 'Email', 'Message', 'interaction_date']
    try:
        with open('interaction_data.csv', 'a', newline='') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=headerscsv)
            # Pass the data in the dictionary as an argument into the writerow() function
            dictwriter_object.writerow(data)
            # Close the file object
            f_object.close()
            return 1
    except Exception as e:
        print('Exception caught : {}'.format(e))
        return 0


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data_ = {'Name': request.form['contact_name'], 'Email': request.form['contact_email'], 'Message': request.form['contact_message'], 'interaction_date': datetime.datetime.now()}
        res = csv_writer(data_)
        if res == 1:
            return render_template("index.html", respon="Data Submitted Successfully.")
        else:
            return render_template("index.html", respon="Some Error Occurred while saving data, Developer has been informed.")
    else:
        return render_template("index.html")


@app.route("/kannada", methods=['GET', 'POST'])
def kannada_pred():
    if request.method == 'POST':
        return render_template('kann.html', result='Submit')
    else:
        return render_template('kann.html')


@app.route("/test")
def test():
    if request.method == 'POST':
        img_topred = request.form['kanimage']
        pred_data = np.array(img_topred)
        pred_data.reshape(pred_data.shape[0], 28, 28, 1)
        pred_data1 = pred_data/255
        model_cnn = load_model('/aimodels/kannada/model_kannada_cnn.h5')
        y_pred = model_cnn.predict_classes(pred_data1)
        print(y_pred)
    else:
        print('else')


if __name__ == "__main__":
    app.run(debug=True, port=8000)
