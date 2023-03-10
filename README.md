# credit_default_probability
La société Home Credit Group a besoin de prédire la probabilité qu'un prêt soit remboursé pour des clients avec peu d'historiques de prêt.   
Nos objectifs sont les suivants:
- Créer un modèle expliquable pour prédire la probabilité de défaut;
- Déployer le modèle dans une API;
- Créer un tableau de bord permettant de visualiser les résultats pour un client.

# Contenu du dépôt
- **preprocessing.ipynb**: Notebook créant la fonction de pré-traitement des données.
- **save_preprocessed.ipynb**: Notebook enregistrant les données pré-traitées sous-forme de csv.
- **model.ipynb**: Notebook testant plusieurs modèles de prédiction et les enregistrant au format joblib.
- **create_api.ipynb**: Notebook créant l'API du modèle retenu avec MLFlow. Pour lancer l'API: `mlflow models serve -m mlflow_model/`.
- **app.py**: Fichier lançant le tableau de bord.

Certains modules python, notamment, doivent être installés pour exécuter les notebooks. Un fichier requirements sera ajouté prochaînement.
