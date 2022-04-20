from flask import Flask, render_template, request  # To handle the web portal requests
from csv import DictWriter  # To write the suggestions in the csv
from python_script.nlp_tasks import nlp_chacha  # Chacha will assist in nlp_tasks
from python_script.cv_tasks import Cv_chacha
import os  # To work with directory
from werkzeug.utils import secure_filename
import datetime  # To play with datetime

app = Flask(__name__)
app.config['upload_ff'] = 'static/uploads'


def csv_writer(data):
    headers_csv = ['Name', 'Email', 'Message', 'interaction_date']
    try:
        with open('interaction_data.csv', 'a', newline='') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=headers_csv)
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
        data_ = {
                    'Name': request.form['contact_name'],
                    'Email': request.form['contact_email'],
                    'Message': request.form['contact_message'],
                    'interaction_date': datetime.datetime.now()
        }
        res = csv_writer(data_)
        if res == 1:
            return render_template("index.html", respon='Data Submitted Successfully.')
        else:
            return render_template("index.html", respon='Some Error Occurred while saving data.')
    else:
        return render_template("index.html")


@app.route("/plant_disease", methods=['GET', 'POST'])
def plant_and_leaf_detection():
    try:
        if request.method == 'POST':
            f = request.files['custom_file']
            if f.filename is ' ':
                return render_template('plant_disease', result=f'No File was provided, Please provide a file to continue')
            else:
                file_path = os.path.join(app.config['upload_ff'], secure_filename(f.filename))
                f.save(file_path)
                cv_helper = Cv_chacha()
                predicted_value = cv_helper.plant_disease(file_path, os.getcwd())
                return render_template('plant_disease.html', result=f'Selected image is of {predicted_value}.')
        else:
            return render_template('plant_disease.html')
    except Exception as exp:
        print(f'Exception Caught {exp}')


@app.route("/review", methods=['GET', 'POST'])
def hotel_review():
    output_dict = {
        0: 'Staff',
        1: 'Food and Beverages',
        2: 'Amenities',
        3: 'Services',
        4: 'Rooms',
        5: 'View and Location',
        6: 'Budget'}
    if request.method == 'POST':
        review_to_predict = request.form['review_text']
        nc = nlp_chacha()
        processed_review = nc.text_preprocessing(review_to_predict)
        predicted_result = nc.hotel_prediction(os.getcwd(), processed_review)
        if predicted_result[0] > 0.6:
            result_stat = 'positive'
        else:
            result_stat = 'negative'
        sort_ed = nc.hotel_reviews(os.getcwd(), processed_review)
        result = []
        for i in sort_ed:
            result.append(output_dict[i])
        return render_template('review.html', op = f'The provided review was {result_stat}', result_op=result[:3])
    else:
        return render_template('review.html')


if __name__ == "__main__":
    app.run(debug=True, port=5001)
