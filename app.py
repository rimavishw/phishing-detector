from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from features import extract_features

app = Flask(__name__)
bundle = joblib.load("model/phishing_model.joblib")
model, columns = bundle["model"], bundle["columns"]

def red_flags(f):
    flags = []
    if f["has_ip"]: flags.append("Uses an IP address instead of a domain name")
    if f["has_at"]: flags.append("Contains '@', which can hide the real destination")
    if not f["uses_https"]: flags.append("Does not use HTTPS")
    if f["url_length"] > 75: flags.append("Unusually long URL")
    if f["num_subdomains"] >= 3: flags.append("Many subdomains")
    if f["suspicious_words"] >= 2: flags.append("Contains several words common in phishing (login, verify, account...)")
    if f["is_shortener"]: flags.append("Uses a URL shortener that hides the destination")
    if f["risky_tld"]: flags.append("Uses a domain ending often abused in phishing")
    if f["num_hyphens_host"] >= 3: flags.append("Many hyphens in the domain name")
    return flags

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url or len(url) > 2000:
        return jsonify(error="Please enter a valid URL (max 2000 characters)."), 400
    f = extract_features(url)
    X = pd.DataFrame([f])[columns]
    prob = float(model.predict_proba(X)[0][1])
    return jsonify(
        phishing_probability=round(prob * 100, 1),
        verdict="Suspicious" if prob >= 0.5 else "Likely safe",
        reasons=red_flags(f),
    )

if __name__ == "__main__":
    app.run(debug=True)