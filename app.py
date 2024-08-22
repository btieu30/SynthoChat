from flask import Flask, render_template
app = Flask(__name__)

#home route, displays the home.html template
@app.route('/')
def display():
    return render_template('home.html')

if __name__ == '__main__':
    app.debug = True
    app.run()