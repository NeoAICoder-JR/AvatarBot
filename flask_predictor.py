from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from difflib import get_close_matches
from flask_cors import CORS

# === INIT APP ===
app = Flask(__name__)
CORS(app)

# === LOAD DATA ===
base_path = r"./"
bio_df = pd.read_csv(f"{base_path}/player_bio.csv")
form_df = pd.read_csv(f"{base_path}/player_form.csv")
surface_elo_df = pd.read_csv(f"{base_path}/surface_specific_elo.csv")

# === VALID PLAYERS SET ===
valid_players = set(bio_df['player']).union(form_df['player']).union(surface_elo_df['Player'])

# === NICKNAME SHORTCUTS ===
nicknames = {
    "djoko": "Novak Djokovic", "djokovic": "Novak Djokovic",
    "nadal": "Rafael Nadal", "rafa": "Rafael Nadal",
    "meddy": "Daniil Medvedev", "medvedev": "Daniil Medvedev",
    "zverev": "Alexander Zverev", "alcaraz": "Carlos Alcaraz",
    "jarry": "Nicolas Jarry", "sinner": "Jannik Sinner",
    "berrettini": "Matteo Berrettini", "fritz": "Taylor Fritz",
    "tsitsipas": "Stefanos Tsitsipas", "dedura": "Diego Dedura",
    "nava": "Emilio Nava"
}

# === COMMON STOPWORDS ===
stopwords = {"who", "will", "win", "between", "on", "and", "the", "a"}

# === PLAYER MATCHING ===
def resolve_player_name(name):
    name = name.lower().strip()
    if not name or len(name) < 2:
        return None
    if name in nicknames:
        return nicknames[name]

    matched = [p for p in valid_players if isinstance(p, str) and name == p.lower()]
    if matched:
        return matched[0]

    if name in stopwords:
        return None

    possible_matches = []
    for p in valid_players:
        if not isinstance(p, str) or not p.strip():
            continue
        try:
            last_name = p.lower().split()[-1]
            if name == last_name:
                possible_matches.append(p)
        except IndexError:
            continue

    if len(possible_matches) == 1:
        return possible_matches[0]
    return None

# === PLAYER EXTRACTION ===
def extract_players_from_text(text):
    text = text.lower()
    candidates = []
    words = text.split()

    for n in range(3, 1, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            if any(word in stopwords for word in phrase.split()):
                continue
            player = resolve_player_name(phrase)
            if player and player not in candidates:
                candidates.append(player)
            if len(candidates) == 2:
                return candidates

    for word in words:
        if word in stopwords:
            continue
        if len(candidates) == 2:
            break
        player = resolve_player_name(word)
        if player and player not in candidates:
            candidates.append(player)

    return candidates

# === PARSE QUESTION ===
def parse_question(question):
    question = question.lower()
    surfaces = ["clay", "hard", "grass", "indoor"]
    surface = next((s for s in surfaces if s in question), None)

    matched_players = extract_players_from_text(question)
    print(f"🧠 Surface: {surface}")
    print(f"👤 Matched Players: {matched_players}")

    if surface and len(matched_players) >= 2:
        return matched_players[0], matched_players[1], surface
    return None, None, None

# === LOAD MODEL ===
model = joblib.load(f"{base_path}/xgboost_surface_model_alpha.pkl")

# === GET STATS ===
def get_player_stats(player_name, surface):
    bio = bio_df[bio_df['player'] == player_name]
    form = form_df[form_df['player'] == player_name]
    elo_row = surface_elo_df[
        (surface_elo_df['Player'] == player_name) &
        (surface_elo_df['Surface'].str.lower() == surface.lower())
    ]

    if bio.empty or form.empty or elo_row.empty:
        print(f"❗ Missing data for: {player_name}")
        return None

    return {
        "age": bio.iloc[0]["age"],
        "height": bio.iloc[0]["height"],
        "recent_form": form.iloc[0]["recent_form"],
        "surface_winrate": form.iloc[0]["surface_winrate"],
        "elo": elo_row.iloc[0]["ELO"]
    }

# === PREDICT FUNCTION ===
def predict_from_players(player_a_raw, player_b_raw, surface):
    stats_a = get_player_stats(player_a_raw, surface)
    stats_b = get_player_stats(player_b_raw, surface)

    print(f"🔍 Comparing '{player_a_raw}' vs '{player_b_raw}' on {surface}")

    if stats_a is None or stats_b is None:
        return f"❌ Stats missing for one of the players: {player_a_raw} or {player_b_raw}"

    features = [
        stats_a["elo"] - stats_b["elo"],
        0,
        stats_a["age"] - stats_b["age"],
        stats_a["height"] - stats_b["height"],
        stats_a["recent_form"] - stats_b["recent_form"],
        stats_a["surface_winrate"] - stats_b["surface_winrate"],
        0
    ]

    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0][pred]
    winner = player_a_raw if pred == 1 else player_b_raw

    return f"🔮 Predicted winner: {winner} (Confidence: {prob:.2%})"

# === FLASK ENDPOINT ===
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        query = data.get("query", "")
        print("📨 Query received:", query)

        p1, p2, surface = parse_question(query)
        if not all([p1, p2, surface]):
            return jsonify({ "answer": "⚠️ Could not understand your question or match surface/players." })

        prediction = predict_from_players(p1, p2, surface)
        print("✅ Prediction:", prediction)
        return jsonify({ "answer": prediction })

    except Exception as e:
        print("💥 Error in /predict:", e)
        return jsonify({ "answer": "❌ Internal server error." }), 500
# === Test ===
@app.route("/")
def home():
    return "Hello from Flask"
    
# === RUN APP ===
if __name__ == "__main__":
    app.run(debug=True)
