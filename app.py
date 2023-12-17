from flask import Flask, request, render_template, url_for
import pandas as pd
import pickle
# import joblib


# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/Recommendation", methods=['GET', 'POST'])
def recommend():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        # model = joblib.load("model.pkl")
        model2 = pickle.load(open('model2.pkl','rb'))
        
        # Get values through input bars
        N = float(request.form.get("N"))
        P = float(request.form.get("P"))
        K = float(request.form.get("K"))
        NS= N/(N+P+K+20)
        PS= P/(N+P+K+20)
        KS= K/(N+P+K+20)
        temperature = request.form.get("temperature")
        humidity = request.form.get("humidity")
        ph = request.form.get("ph")
        rainfall = request.form.get("rainfall")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[NS, PS, KS, temperature, humidity, ph,  rainfall]], columns = ["NS", "PS", "KS", "temperature", "humidity", "ph", "rainfall"])
        
        # Get prediction
        prediction = model2.predict(X)[0]
    else:
        prediction = ""
        data=""
        
    return render_template("crop recommendation.html", output = prediction)

@app.route("/Govermentskim")
def schemes( ):
    return render_template("Govermentskim.html")

@app.route("/CropPrices")
def prices( ):
    return render_template("Crop prices.html")

@app.route("/CropPrices/Forecast")
def forecast( ):
    return render_template("index test2.html")

@app.route("/Weather")
def weather( ):
    return render_template("Weather.html")

@app.route("/Apple")
def apple( ):
    return render_template("Apple.html")

@app.route("/Banana")
def banana( ):
    return render_template("banana.html")

@app.route("/Blackgram")
def blackgram( ):
    return render_template("blackgram.html")

@app.route("/Chickpea")
def chickpea( ):
    return render_template("Chickpea.html")

@app.route("/Coconut")
def coconut( ):
    return render_template("Coconut.html")

@app.route("/Coffee")
def coffee( ):
    return render_template("Coffee.html")

@app.route("/Cotton")
def cotton( ):
    return render_template("cotton.html")

@app.route("/Grapes")
def grapes( ):
    return render_template("grapes.html")

@app.route("/Jute")
def jute( ):
    return render_template("Jute.html")

@app.route("/Kidneybeans")
def kidneybeans( ):
    return render_template("kidneybeans.html")

@app.route("/Lentil")
def lentil( ):
    return render_template("lentil.html")

@app.route("/Maize")
def maize( ):
    return render_template("Maize.html")

@app.route("/Mango")
def mango( ):
    return render_template("mango.html")

@app.route("/Mothbeans")
def mothbeans( ):
    return render_template("Mothbean.html")

@app.route("/Mungbean")
def mungbean( ):
    return render_template("greengram.html")

@app.route("/Muskmelon")
def muskmelon( ):
    return render_template("muskmelon.html")

@app.route("/Orange")
def orange( ):
    return render_template("orange.html")

@app.route("/Papaya")
def papaya( ):
    return render_template("papaya.html")

@app.route("/Pigeonpeas")
def pigeonpeas( ):
    return render_template("pigeonpea.html")

@app.route("/Pomegranate")
def pomegranate( ):
    return render_template("pomegrante.html")

@app.route("/Rice")
def rice( ):
    return render_template("Rice.html")

@app.route("/Watermelon")
def watermelon( ):
    return render_template("watermelon.html")

# Running the app
if __name__ == '__main__':
	# app.jinja_env.auto_reload = True
	# app.config['TEMPLATES_AUTO_RELOAD'] = True
	# app.jinja_env.cache = None
	app.run(debug = True)
