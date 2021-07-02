from flask import Flask, request
from flask.templating import render_template
from infere import pipleline
from infere_bert import pipleline_bertbase
app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/", methods=['GET', 'POST'])
def form_post():
    text = request.form['inputtext']

    # result = pipleline(text)
    result = pipleline_bertbase(text)

    return render_template("submit.html", raw=text, result=result)

if __name__ == '__main__':
    app.run(debug=True)