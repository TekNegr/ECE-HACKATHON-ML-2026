
import pandas as pd

def predict_workout(sport, speed, duration, model=None):
    # Vérification des données
    if not isinstance(sport, str) or not sport.strip():
        raise ValueError("Le sport doit être renseigné.")

    speed = float(speed)
    duration = float(duration)

    if speed < 0 or duration <= 0:
        raise ValueError("Vitesse ou durée invalide.")

    # Préparation des données
    data = pd.DataFrame([{
        "sport": sport,
        "speed": speed,
        "duration": duration
    }])

    # Simulation en attendant le modèle
    if model is None:
        return "moderate"

    # Prédiction réelle
    return model.predict(data)[0]


print(predict_workout("running", 10, 30))
