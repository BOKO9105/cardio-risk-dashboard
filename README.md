# Projet de Prédiction du Risque Cardiométabolique (Ten Year CHD)

Bienvenue dans l'application compagnon conçue pour analyser, modéliser et visualiser le risque de maladie coronarienne (CHD) sur 10 ans ! Ce projet découle de l'analyse menée dans le carnet de notes Python `Analyse_boko.ipynb`.

## Structure du Projet

- `data/` : Contient le jeu de données local (train.csv).
- `models/` : Dossier généré contenant le meilleur classificateur exporté (`best_model.pkl`) et le rapport des modèles (`model_results.json`).
- `train_model.py` : Script backend pour nettoyer, comparer et exporter les modèles de Machine Learning.
- `app.py` : Frontend pour exploiter les résultats (le tableau de bord en direct).
- `requirements.txt` / `environment.yml` : Liste des dépendances nécessaires au format pip et conda.

## Instructions d'installation en local

1. **Création de l'Environnement (Si vous préférez Anaconda/Conda) :**
   ```bash
   conda env create -f environment.yml
   conda activate ml_env
   pip install -r requirements.txt
   ```

2. **Génération du Modèle Backend :**
   Exécutez la logique d'entraînement décrite dans le script. À la fin, le meilleur algorithme sera sélectionné selon le score F1 pour contrecarrer le déséquilibre des classes.
   ```bash
   python train_model.py
   ```

3. **Lancement du Tableau de Bord (Streamlit) :**
   Visualisez vos clusters, statistiques et l'outil de prédiction interactif dans votre navigateur internet en local.
   ```bash
   streamlit run app.py
   ```

## Déploiement sur GitHub & Streamlit Community Cloud

Ce projet est déjà paramétré avec son `.gitignore` et ses librairies dans le `requirements.txt`.
Pour rendre l'analyse publique et la montrer aux recruteurs, il vous suffit de :
1. Pousser `git push` ce répertoire sur **GitHub**. 
   - _Rappel_ : Vérifiez que github a le script `train_model.py` et le modèle sauvegardé dans `models/best_model.pkl`. 
   - _Si votre modèle dépasse 100 Mo_, servez-vous de `git lfs`. Le notre est tout petit! 
2. Allez sur **[Streamlit Community Cloud](https://share.streamlit.io/)**, liez votre compte GitHub.
3. Paramétrez `app.py` comme chemin principal ("Main file path") et lancez le déploiement gratuit!
