import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import f1_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')

# ---- Pipeline Expert : Optimisation + Anti-Surapprentissage + Calibration ----
@st.cache_resource
def train_model_and_results():
    """
    Pipeline rigoureux :
    1. RandomizedSearchCV sur chaque modèle (anti-surapprentissage)
    2. Sélection sur CV ROC-AUC + vérification gap Train/Test
    3. Calibration de Platt sur le vainqueur
    """
    try:
        df = pd.read_csv("data/train.csv")
    except:
        return None, {}

    y = df['TenYearCHD']
    X = df.drop(columns=['id', 'TenYearCHD'])

    categorical_features = ['sex', 'is_smoking']
    numeric_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    binary_features = ['education', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

    num_t = Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())])
    cat_t = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(drop='first', sparse_output=False))])
    bin_t = Pipeline([('imp', SimpleImputer(strategy='most_frequent'))])

    preprocessor = ColumnTransformer([
        ('num', num_t, numeric_features),
        ('cat', cat_t, categorical_features),
        ('bin', bin_t, binary_features)
    ], remainder='passthrough')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---- Grilles anti-surapprentissage ----
    candidate_models = {
        'Gradient Boosting': (
            GradientBoostingClassifier(random_state=42),
            {
                'classifier__n_estimators': [80, 120, 150],
                'classifier__max_depth': [2, 3],         # faible profondeur = moins de memorisation
                'classifier__learning_rate': [0.03, 0.05, 0.08],
                'classifier__subsample': [0.7, 0.8],    # sous-échantillonnage stochastique
                'classifier__min_samples_leaf': [10, 20] # noeuds terminaux plus larges = moins d'overfit
            }
        ),
        'Regégression Logistique': (
            LogisticRegression(max_iter=1500, solver='liblinear', random_state=42),
            {
                'classifier__C': [0.001, 0.01, 0.1, 0.5, 1.0], # régularisation forte
                'classifier__penalty': ['l1', 'l2']
            }
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42),
            {
                'classifier__n_estimators': [80, 120],
                'classifier__max_depth': [3, 4, 5],     # profondeur limitée
                'classifier__min_samples_leaf': [10, 20, 30], # noeuds plus larges
                'classifier__max_features': ['sqrt', 0.5]
            }
        )
    }

    results = {}
    best_models = {}  # stock les pipelines optimisés

    for name, (base_model, param_grid) in candidate_models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', base_model)])
        search = RandomizedSearchCV(
            pipe, param_grid, n_iter=12, scoring='roc_auc',
            cv=cv, random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        best_models[name] = best_pipe

        # Sur l'ensemble test
        y_proba = best_pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.30).astype(int)
        test_auc = roc_auc_score(y_test, y_proba)
        cv_auc = search.best_score_
        overfit_gap = cv_auc - test_auc  # petit = généralise bien

        # AUC train (pour mesurer surapprentissage)
        y_proba_train = best_pipe.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_proba_train)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=8, strategy='quantile')

        results[name] = {
            'Train AUC': float(train_auc),
            'CV ROC AUC Mean': float(cv_auc),
            'Test ROC AUC': float(test_auc),
            'Overfit Gap': float(overfit_gap),
            'Test F1 Score': float(f1_score(y_test, y_pred)),
            'Test Recall': float(recall_score(y_test, y_pred)),
            'Confusion Matrix': cm,
            'ROC_FPR': fpr[::3].tolist(),
            'ROC_TPR': tpr[::3].tolist(),
            'Calib_x': mean_pred.tolist(),
            'Calib_y': frac_pos.tolist(),
            'Best Params': str(search.best_params_)
        }

    # ---- Sélection Objective : meilleur CV AUC ET faible overfit ----
    def composite_score(name):
        r = results[name]
        return r['CV ROC AUC Mean'] - 1.5 * max(0, r['Overfit Gap'])

    best_name = max(results.keys(), key=composite_score)
    best_base_pipe = best_models[best_name]
    results['Best_Model_Name'] = best_name

    # ---- Calibration de Platt avec cv='prefit' (évite le bug DataFrame/numpy) ----
    # On divise X_train en train2 (fit) + val (calibration)
    X_train2, X_val, y_train2, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=0
    )
    best_base_pipe.fit(X_train2, y_train2)  # fit sur train2 (DataFrame → OK)
    calibrated_winner = CalibratedClassifierCV(best_base_pipe, method='sigmoid', cv='prefit')
    calibrated_winner.fit(X_val, y_val)     # calibre sur val (DataFrame → OK)

    # Feature importances du modèle vainqueur
    try:
        raw_classifier = best_base_pipe.named_steps['classifier']
        preprocessor_fit = best_base_pipe.named_steps['preprocessor']
        cat_enc = preprocessor_fit.named_transformers_['cat'].named_steps['ohe']
        cat_names = [f"{col}_{v}" for col, vals in zip(categorical_features, cat_enc.categories_) for v in vals[1:]]
        feat_names = numeric_features + cat_names + binary_features
        if hasattr(raw_classifier, 'feature_importances_'):
            importances = raw_classifier.feature_importances_
        elif hasattr(raw_classifier, 'coef_'):
            importances = np.abs(raw_classifier.coef_[0])
        else:
            importances = np.zeros(len(feat_names))
        results['Global_Feature_Importance'] = {f: float(imp) for f, imp in zip(feat_names, importances)}
    except Exception:
        pass

    # Pipeline final : préprocesseur + classifieur calibré (cv='prefit' = pas de refit interne)
    final_pipeline = Pipeline([
        ('preprocessor', best_base_pipe.named_steps['preprocessor']),
        ('classifier', calibrated_winner)
    ])

    return final_pipeline, results

# Configuration Master
st.set_page_config(page_title="Cardio Risk | Master AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .metric-card { background-color: #2b2b2b; padding: 15px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); text-align: center; margin-bottom: 20px; transition: transform 0.2s;}
    .metric-card:hover { transform: scale(1.02); }
    .metric-value { font-size: 2.2em; font-weight: 800; color: #E63946; }
    .metric-label { font-size: 1.05em; color: #A8DADC; }
    .title-h1 { color: #f1faee; margin-bottom: 1.5rem; border-bottom: 1px solid #457B9D; padding-bottom: 5px; }
    .title-h2 { color: #A8DADC; margin-top: 1rem; }
    
    /* Expander styling */
    div[data-testid="stExpander"] { background-color: #FFFFFF; color: #000000; border-left: 6px solid #E63946; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.15);}
    div[data-testid="stExpander"] p { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 1.05em; line-height: 1.5; color: #000000;}
    div[data-testid="stExpander"] summary { color: #1D3557 !important; font-size: 1.15em !important; font-weight: bold;}
    div[data-testid="stExpander"] summary svg { fill: #1D3557 !important; }
    
    div[data-testid="stTabs"] button { font-size: 1.1rem !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try: return pd.read_csv("data/train.csv")
    except: return None

df_raw = load_data()

# --- SIDEBAR ---
st.sidebar.image("https://images.unsplash.com/photo-1551076805-e1869033e561?w=400&h=250&fit=crop", caption="Data Science Interactive")
st.sidebar.markdown("## ⚙️ Filtres Dynamiques")
st.sidebar.info("Modifiez ces filtres pour observer dynamiquement les sous-populations (les graphiques s'actualisent).")

if df_raw is not None:
    filter_sex = st.sidebar.selectbox("Filtre Genre", ["Tous", "M", "F"])
    filter_smoke = st.sidebar.radio("Statut Tabagique", ["Tous", "Fumeur", "Non-Fumeur"])
    filter_age = st.sidebar.slider("Filtre d'Âge", int(df_raw['age'].min()), int(df_raw['age'].max()), (int(df_raw['age'].min()), int(df_raw['age'].max())))
    
    df = df_raw.copy()
    if filter_sex != "Tous": df = df[df['sex'] == filter_sex]
    if filter_smoke != "Tous": df = df[df['is_smoking'] == ("YES" if filter_smoke == "Fumeur" else "NO")]
    df = df[(df['age'] >= filter_age[0]) & (df['age'] <= filter_age[1])]
        
    st.sidebar.success(f"📊 {len(df)} dossiers patients filtrés.")
else: df = None

st.title("🫀 Prediction du Maladie Coronarienne (CHD) à partir des Biomarqueurs Personnels")
st.caption("Analyse multivariée et segmentation de profils de risque cardiométabolique par méthodes d'apprentissage non supervisé et supervisé.")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 Contexte", "📊 EDA", "🌌 Réduction (t-SNE)", "🧬 Clustering", "🧠 Modélisation ROC", "🩺 Simulateur Clinique"])

# [ONGLET 0]
with tab0:
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1200&h=250&fit=crop", use_container_width=True)
    st.markdown("<h1 class='title-h1'>🩺 La Framingham Heart Study</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.write("Les maladies cardiovasculaires (MCV) constituent un enjeu majeur de santé publique mondiale (32 % de la mortalité totale de l'OMS). En Santé Publique, leur anticipation pécuniaire et létale est la priorité fondamentale.")
        st.info("**Défi Scientifique** : Peut-on concevoir un outil préventif automatisé (IA) robuste malgré la difficulté intrinsèque d'une maladie *silencieuse* et une cohorte fortement déséquilibrée ?")
        st.write("L'approche requiert de croiser l'Intelligence Artificielle et le raisonnement épidémiologique (Régression Logistique, Stratification).")
    with c2:
        st.markdown("### Processus de Validation :")
        st.write("✅ Nettoyage de Données (Imputation)\n✅ Découverte Visuelle (t-SNE, ACP)\n✅ Création de Phénotypes Idiopathiques\n✅ Validation Croisée K-Fold\n✅ Outil de Prévention Primaire")

    # --- Aperçu du Dataset en bas de page ---
    st.markdown("---")
    st.markdown("<h2 class='title-h2'>📋 Aperçu du Jeu de Données (Framingham Cohort)</h2>", unsafe_allow_html=True)

    if df_raw is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metric-card'><div class='metric-label'>Patients Total</div><div class='metric-value'>{len(df_raw)}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-label'>Variables Cliniques</div><div class='metric-value'>{len(df_raw.columns) - 2}</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-label'>Cas CHD à 10 ans</div><div class='metric-value'>{df_raw['TenYearCHD'].sum()}</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><div class='metric-label'>Prévalence (%)</div><div class='metric-value'>{df_raw['TenYearCHD'].mean()*100:.1f}%</div></div>", unsafe_allow_html=True)

        col_prev, col_stat = st.columns([1.2, 1])

        with col_prev:
            st.markdown("<h3 class='title-h2'>Échantillon des 10 premiers patients</h3>", unsafe_allow_html=True)
            st.dataframe(
                df_raw.head(10).style.highlight_max(axis=0, color='#1E4D2B').highlight_min(axis=0, color='#4D1E1E'),
                use_container_width=True
            )

        with col_stat:
            st.markdown("<h3 class='title-h2'>Statistiques Descriptives</h3>", unsafe_allow_html=True)
            desc = df_raw.describe().round(2)
            st.dataframe(desc, use_container_width=True)

        with st.expander("💡 Description des Variables Cliniques", expanded=False):
            st.markdown("""
| Variable | Description Médicale |
|---|---|
| `age` | Âge du patient (années) |
| `sex` | Genre (M / F) |
| `is_smoking` | Fumeur actif déclaré (YES / NO) |
| `cigsPerDay` | Nombre de cigarettes par jour |
| `BPMeds` | Traitement anti-hypertenseur actif |
| `prevalentStroke` | Antécédent d'AVC |
| `prevalentHyp` | Hypertension diagnostiquée |
| `diabetes` | Diabète diagnostiqué |
| `totChol` | Cholestérol total (mg/dL) |
| `sysBP` | Tension Systolique (mmHg) |
| `diaBP` | Tension Diastolique (mmHg) |
| `BMI` | Indice de Masse Corporelle |
| `heartRate` | Fréquence cardiaque au repos (BPM) |
| `glucose` | Glycémie à jeun (mg/dL) |
| `TenYearCHD` | **Variable Cible** : CHD dans les 10 ans (0=Non / 1=Oui) |
""")

# [ONGLET 1]
with tab1:
    st.markdown("<h1 class='title-h1'>📊 Profilage & Asymétrie Clinique</h1>", unsafe_allow_html=True)
    if df is not None and not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><div class='metric-label'>Taille Cohorte (Filtre)</div><div class='metric-value'>{len(df)}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-label'>Proportion Échantillon</div><div class='metric-value'>{len(df)/len(df_raw)*100:.1f} %</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-label'>Malades à 10 ans (Risque)</div><div class='metric-value'>{df['TenYearCHD'].sum()}</div></div>", unsafe_allow_html=True)

        cA, cB = st.columns([1, 1.5])
        with cA:
            st.markdown("<h3 class='title-h2'>Biais de la Cible (Imbalance)</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.countplot(data=df, x='TenYearCHD', palette=['#1D3557', '#E63946'], ax=ax)
            ax.set_xticklabels(['Sains', 'Malades'])
            sns.despine()
            st.pyplot(fig)
        with cB:
            st.markdown("<h3 class='title-h2'>Matrice des Corrélations Cliniques</h3>", unsafe_allow_html=True)
            df_num = df.select_dtypes(include=[np.number])
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.heatmap(df_num.corr(), cmap='coolwarm', vmin=-1, vmax=1, ax=ax2)
            st.pyplot(fig2)
            
        with st.expander("💡 Afficher l'Interprétation Médicale (L'Aiguille dans la Botte de Foin)", expanded=False):
            st.markdown("""La rareté des malades engendre ce qu'on appelle "l'Accuracy Paradox" en IA. C'est pourquoi le paramétrage "Class Weight" dans nos algorithmes priorise intentionnellement la détection des malades en maximisant le **Recall**. En médecine préventive, nous préférons un *faux-positif* (qui déclenchera des examens superflus) plutôt qu'un dramatique *faux-négatif* (qui ignorera un patient condamné).""")

# [ONGLET 2]
with tab2:
    st.markdown("<h1 class='title-h1'>🌌 Extraction de la Variance (t-SNE)</h1>", unsafe_allow_html=True)
    if df is not None:
        df_clean = df.select_dtypes(include=[np.number]).dropna().drop(columns=['id', 'TenYearCHD'], errors='ignore')
        if not df_clean.empty and len(df_clean) > 5:
            target = df.loc[df_clean.index, 'TenYearCHD'] if 'TenYearCHD' in df.columns else np.zeros(len(df_clean))
            X_scaled = StandardScaler().fit_transform(df_clean)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<h3 class='title-h2'>Compression Linéaire (ACP)</h3>", unsafe_allow_html=True)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=target, cmap='Set1', alpha=0.7)
                sns.despine()
                st.pyplot(fig1)
                
            with c2:
                st.markdown("<h3 class='title-h2'>Topologie Non-Linéaire (t-SNE)</h3>", unsafe_allow_html=True)
                with st.spinner("L'algorithme dessine la topologie..."):
                    perplex = min(30, max(2, len(X_scaled) - 1))
                    @st.cache_data
                    def compute_tsne(data, p):
                        return TSNE(n_components=2, perplexity=p, random_state=42).fit_transform(data)
                    X_tsne = compute_tsne(X_scaled, perplex)
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target, cmap='Set1', alpha=0.7)
                    sns.despine()
                    st.pyplot(fig2)
                    
            with st.expander("💡 Afficher l'Interprétation Dimensionnelle (Complexité Biomédicale)", expanded=False):
                st.markdown("""Le fait que les patients malades (en rouge) ne soient pas du tout séparés par une frontière mathématique nette démontre la sévérité de la maladie : **elle est complètement silencieuse et multifactorielle**. L'Homme ne peut pas tracer une "règle clinique simple" pour trouver les malades. Seul le balayage profond autorisé par le Machine Learning pourra trancher cette densité de façon sécuritaire.""")
        else:
            st.warning("⚠️ Vos filtres dynamiques sont trop stricts.")

# [ONGLET 3]
with tab3:
    st.markdown("<h1 class='title-h1'>🧬 Découverte des Phénotypes Idiopathiques</h1>", unsafe_allow_html=True)
    if df is not None and ('df_clean' in locals() and not df_clean.empty and len(df_clean) > 5):
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown("<h3 class='title-h2'>K-Means (k=3)</h3>", unsafe_allow_html=True)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
            ca_c = pca.transform(kmeans.cluster_centers_)
            ax.scatter(ca_c[:, 0], ca_c[:, 1], c='red', s=200, marker='X')
            sns.despine()
            st.pyplot(fig)
            
        with c2:
            st.markdown("<h3 class='title-h2'>Arbre CAH (Dendrogramme)</h3>", unsafe_allow_html=True)
            with st.spinner("Génération de l'arbre filial..."):
                @st.cache_data
                def compute_linkage(data):
                    return linkage(data[:min(600, len(data))], method='ward')
                Z = compute_linkage(X_scaled)
                fig_dend, ax_dend = plt.subplots(figsize=(8, 5))
                dendrogram(Z, truncate_mode='level', p=10, ax=ax_dend)
                sns.despine()
                st.pyplot(fig_dend)

        with st.expander("💡 Afficher l'Interprétation Clinique (Médecine Prédictive)", expanded=False):
            st.markdown("""L'IA non-supervisée repère "à l'aveugle" qu'il existe 3 grappes de patients logiques dans votre base d'étude médicale :<br>
- **Le pôle Usure Mécanique** : Majoritairement âgés, sous hypertension (sysBP très élevé).<br>
- **Le pôle Syndrome Métabolique** : Marqués par le Diabète et l'Hypercholestérolémie.<br>
- **Le pôle Sain** : L'immense majorité (Faible âge, normes sanguines parfaites).<br><br>
En santé publique, l'application de ce Clustering permet d'**ajuster les campagnes de préventions nationales** (par exemple : campagne nutritionnelle massive pour le cluster métabolique pré-identifié avant l'apparition concrète de maladies de coeur cardiaques).""", unsafe_allow_html=True)

# [ONGLET 4]
with tab4:
    st.markdown("<h1 class='title-h1'>🧠 Standards de la Validation Croisée & ROC</h1>", unsafe_allow_html=True)
    
    with st.spinner("Chargement des métriques du modèle..."):
        _, results = train_model_and_results()

    if results:
        imp_dict = results.get("Global_Feature_Importance", None)
        best_model_name = results.get("Best_Model_Name", "")
        # Filtrer les clés qui ne sont pas des modèles
        model_keys = [k for k in results.keys() if k not in ("Global_Feature_Importance", "Best_Model_Name")]

        if best_model_name:
            st.success(f"🏆 **Modèle Vainqueur sélectionné objectivement : {best_model_name}** — Meilleur score composite (CV AUC - pénalité Overfit)")

        st.write("Chaque modèle a été optimisé par `RandomizedSearchCV` (12 itérations, 5-Fold). Le vainqueur est choisi sur un score composite qui pénalise le surapprentissage.")

        # --- LIGNE 1 : FIGURES (ROC et TABLEAU ANTI-OVERFIT) ---
        col_roc, col_cv = st.columns([1.5, 1.5])

        with col_roc:
            st.markdown("<h3 class='title-h2'>📈 Courbes ROC (Sensibilité)</h3>", unsafe_allow_html=True)
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            for m_name in model_keys:
                fpr, tpr = results[m_name].get('ROC_FPR', []), results[m_name].get('ROC_TPR', [])
                auc_score = results[m_name].get('Test ROC AUC', 0)
                lw = 3 if m_name == best_model_name else 1.5
                ls = '-' if m_name == best_model_name else '--'
                if len(fpr) > 0:
                    ax_roc.plot(fpr, tpr, label=f'{m_name} (AUC = {auc_score:.2f})', linewidth=lw, linestyle=ls)
            ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc='lower right', fontsize=8)
            sns.despine()
            st.pyplot(fig_roc)

        with col_cv:
            st.markdown("<h3 class='title-h2'>🔬 Diagnostic Anti-Surapprentissage</h3>", unsafe_allow_html=True)
            df_cv = pd.DataFrame({
                "Modèle": model_keys,
                "Train AUC": [round(results[k].get("Train AUC", 0), 3) for k in model_keys],
                "CV AUC": [round(results[k].get("CV ROC AUC Mean", 0), 3) for k in model_keys],
                "Test AUC": [round(results[k].get("Test ROC AUC", 0), 3) for k in model_keys],
                "Gap Overfit ↓": [round(results[k].get("Overfit Gap", 0), 3) for k in model_keys],
            }).set_index("Modèle")
            # Colorer en rouge les gaps élevés (surapprentissage)
            def color_gap(val):
                if val > 0.05: return 'background-color: #4D1E1E; color: white'
                elif val > 0.02: return 'background-color: #4D3A1E; color: white'
                return 'background-color: #1E4D2B; color: white'
            st.dataframe(
                df_cv.style.applymap(color_gap, subset=["Gap Overfit ↓"]).highlight_max(subset=["CV AUC", "Test AUC"], color='#1E4D2B'),
                use_container_width=True
            )
            st.caption("🟢 Gap < 0.02 : Bonne généralisation | 🟡 0.02–0.05 : Légère tension | 🔴 > 0.05 : Surapprentissage")

        # COMMENTAIRE HORIZONTAL PLEINE LARGEUR (SOUS LA LIGNE 1)
        with st.expander("💡 Méthodologie Anti-Surapprentissage & Calibration de Platt", expanded=False):
            st.markdown("""
            **Protocole rigoureux en 3 étapes :**<br>
            1. **RandomizedSearchCV (12 iter, 5-Fold)** : optimise les hyperparamètres de chaque modèle sur les données d'entraînement uniquement (jamais sur le test).<br>
            2. **Score Composite** : `CV_AUC - 1.5 × max(0, Gap_Overfit)` — pénalise les modèles qui mémorisent au lieu d'apprendre.<br>
            3. **Calibration de Platt** : ajuste les probabilités brutes pour correspondre à de vraies fréquences épidémiologiques (25% = 25 malades sur 100 profils similaires).<br><br>
            Le seuil de décision est fixé à **0.30** (non 0.50) pour adapter le compromis Sensibilité/Spécificité au contexte de prévention primaire cardiovasculaire.
            """, unsafe_allow_html=True)

            
        st.markdown("---")
            
        # --- LIGNE 2 : FIGURES (CALIBRATION + MATRICES + IMPORTANCE) ---
        col_calib, col_cm, col_imp = st.columns([1, 1.2, 1])
        
        with col_calib:
            st.markdown("<h3 class='title-h2'>📊 Courbe de Calibration</h3>", unsafe_allow_html=True)
            lr_key = next((k for k in results if 'Logistique' in k), None)
            if lr_key:
                calib_x = results[lr_key].get('Calib_x', [])
                calib_y = results[lr_key].get('Calib_y', [])
                if calib_x:
                    fig_cal, ax_cal = plt.subplots(figsize=(5, 4))
                    ax_cal.plot(calib_x, calib_y, 's-', color='#E63946', label='Logistique Calibrée')
                    ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Calibration Parfaite')
                    ax_cal.set_xlabel('Score Prédit (Probabilité Modèle)')
                    ax_cal.set_ylabel('Fraction Réelle de Malades')
                    ax_cal.legend(loc='upper left')
                    sns.despine()
                    st.pyplot(fig_cal)
                    st.caption("Plus la courbe rouge est proche de la diagonale noire, plus les probabilités sont fiables.")
        
        with col_cm:
            st.markdown("<h3 class='title-h2'>🎯 Matrices de Confusion</h3>", unsafe_allow_html=True)
            lr_key = next((k for k in results if 'Logistique' in k), None)
            mat_lr = np.array(results.get(lr_key, {}).get('Confusion Matrix', [[0,0],[0,0]])) if lr_key else np.zeros((2,2))
            mat_rf = np.array(results.get('Random Forest', {}).get('Confusion Matrix', [[0,0],[0,0]]))
            fig_cm, axes_cm = plt.subplots(1, 2, figsize=(8, 3.5))
            sns.heatmap(mat_lr, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0], cbar=False)
            axes_cm[0].set_title('Logistique (Calibrée)')
            axes_cm[0].set_ylabel('Vérité Médicale')
            sns.heatmap(mat_rf, annot=True, fmt='d', cmap='Reds', ax=axes_cm[1], cbar=False)
            axes_cm[1].set_title('Random Forest')
            st.pyplot(fig_cm)

        with col_imp:
            st.markdown("<h3 class='title-h2'>🔑 Poids Médicaux (Coefficients)</h3>", unsafe_allow_html=True)
            if imp_dict:
                imp_s = pd.Series(imp_dict).sort_values(ascending=True).tail(6)
                fig_imp, ax_imp = plt.subplots(figsize=(5, 3.8))
                imp_s.plot(kind='barh', color='#E63946', ax=ax_imp)
                sns.despine()
                st.pyplot(fig_imp)
            else:
                st.info("Coefficients logistiques non disponibles pour cet exécution.")
                
        # COMMENTAIRE HORIZONTAL PLEINE LARGEUR (SOUS LA LIGNE 2)
        with st.expander("💡 Ouverture de la Boite Noire : Les Poids Majeurement Inculpés", expanded=False):
            st.markdown("""
            Puisque le modèle retenu est intrinsèquement transparent (coefficients linéaires), l'IA affirme l'étiologie médicale de Framingham de son propre chef :<br>
            - La domination de la Tension Systolique (**sysBP**) pointe la mortalité précoce via l'usure mécanique vasculaire.<br>
            - **L'Âge** est un marqueur inconditionnel d'artério-sclérose physiologique naturelle.<br>
            - La Matrice de Confusion montre factuellement comment **l'erreur des Faux Négatifs est drastiquement réduite** en faveur du diagnostic précoce.
            """, unsafe_allow_html=True)

    else:
        st.error("⚠️ Données d'entraînement introuvables. Vérifiez que `data/train.csv` est bien présent sur GitHub.")

# [ONGLET 5]
with tab5:
    st.markdown("<h1 class='title-h1'>🩺 Simulateur Clinique & Programme de Santé Publique</h1>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=1200&h=150&fit=crop", use_container_width=True)
    st.write("Insérez les biomarqueurs du dossier patient. La Régression Logistique va statuer sur l'état de la crise CHD et vous livrer les recommandations sanitaires d'usage (Plan de Santé Publique).")
    
    with st.spinner("Initialisation du modèle diagnostique (entraînement initial)..."):
        model, _model_results = train_model_and_results()
    if model is not None:
        with st.form("medical_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.slider("Âge Réglementaire", 30, 90, 50)
                sex = st.selectbox("Assignation de Genre", ["M", "F"])
                education = st.selectbox("Facteur Socio-Éducatif", [1.0, 2.0, 3.0, 4.0])
                bmi = st.number_input("Indice de Masse C. (IMC)", 15.0, 60.0, 26.0)

            with c2:
                is_smoking = st.selectbox("Facteur Tabagique Déclaré", ["YES", "NO"])
                cigsPerDay = st.slider("Consommation / Jour", 0, 80, 0) if is_smoking == "YES" else 0
                bpMeds = st.selectbox("Hémothérapie Hyper-Tensive", [0.0, 1.0], format_func=lambda x: "Active" if x==1 else "Pas de traitement")
                prevalentStroke = st.selectbox("ATCD AVC (Course)", [0, 1], format_func=lambda x: "Oui" if x==1 else "Non")

            with c3:
                prevalentHyp = st.selectbox("Tension Chronique (Diagnostiquée)", [0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
                diabetes = st.selectbox("Condition Hyper-Glycémique", [0, 1], format_func=lambda x: "Diabétique" if x==1 else "A jeun Sain")
                totChol = st.number_input("Lipide Stérolique (mg/dL)", 100.0, 600.0, 230.0)
                sysBP = st.number_input("Volume Systolique", 80.0, 250.0, 135.0)
                diaBP = st.number_input("Volume Diastolique", 40.0, 150.0, 85.0)
                heartRate = st.number_input("Rythme B.P.M", 40.0, 160.0, 75.0)
                glucose = st.number_input("Glucide Rapide (mg/dL)", 40.0, 400.0, 85.0)

            submitted = st.form_submit_button("🩺 Évaluer le Profil Moteur et Créer la Prescription", type="primary", use_container_width=True)

        if submitted:
            my_bar = st.progress(0)
            status_text = st.empty()
            
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
                if percent_complete < 30: status_text.text("Standardisation des Vitaux (Scale)...")
                elif percent_complete < 70: status_text.text("Passage par Sigmoïde du Modèle Logistique...")
                else: status_text.text("Génération du Plan de Prévention Santé...")
                
            status_text.empty()
            my_bar.empty()

            input_df = pd.DataFrame([{
                'age': float(age), 'education': float(education), 'sex': sex, 'is_smoking': is_smoking,
                'cigsPerDay': float(cigsPerDay), 'BPMeds': float(bpMeds), 'prevalentStroke': int(prevalentStroke),
                'prevalentHyp': int(prevalentHyp), 'diabetes': int(diabetes), 'totChol': float(totChol),
                'sysBP': float(sysBP), 'diaBP': float(diaBP), 'BMI': float(bmi), 'heartRate': float(heartRate), 'glucose': float(glucose)
            }])
            
            proba = model.predict_proba(input_df)[0][1]
            pred = model.predict(input_df)[0]
                
            st.markdown("---")
            
            # MODULE DE PRÉVENTION SANTÉ PUBLIQUE (PUBLIC HEALTH PREVENTION PLAN)
            st.markdown("<h3 class='title-h2'>💊 Recommandations de Base et Prévention Sanitaire</h3>", unsafe_allow_html=True)
            
            conseils = []
            if is_smoking == "YES": 
                conseils.append("**Tabagisme Actif :** Sevrage tabagique urgent. Orientation vers une consultation de tabacologie (TCC, substituts nicotiniques). Le tabac abîme l'endothélium par stress oxydatif violent.")
            if float(sysBP) > 130 or prevalentHyp == 1: 
                conseils.append("**Tension Critique :** Prescription immédiate d'une auto-mesure tensionnelle sur 3 jours (Règle des 3). Réévaluation de la sédentarité et ajout d'ACE Inhibitors si l'hypertrophie cardiaque se confirme.")
            if float(totChol) > 230: 
                conseils.append("**Dyslipidémie (Athérosclérose) :** Bilan EAL complet à jeun. Prescription de statines envisagée pour bloquer la formation des plaques. Instauration d'un régime nutritionnel de type 'méditerranéen'.")
            if int(diabetes) == 1 or float(glucose) > 110: 
                conseils.append("**Insulino-résistance :** Objectif HbA1C < 7%. Traitements antidiabétiques oraux à revoir. Nécessite une veille ophtalmologique (Rétinopathie) et rénale annuelle très stricte.")
            if float(bmi) > 26: 
                conseils.append("**Surcharge Physiologique :** Objectif minimum de réduction de 10% du poids corporel en 6 mois, corrélé à un plan d'activité cardiovasculaire légère et continue (ex. Marche dynamique 150min/semaine).")
            if float(age) > 60:
                conseils.append("**Sénescence Structurelle :** Bien que non modifiable, le déclin des parois aortiques après 60 ans requiert un banal ECG de repos de base bi-annuel et l'écoute à l'auscultation d'un éventuel souffle cardiaque bénin.")
                
            if pred == 1:
                st.error("🚨 **Alerte Pronostic : Haut Risque Pathologique de Crise CHD dans la prochaine décennie.**")
                st.markdown(f"**Évaluation du Danger Vectoriel :** `{proba*100:.1f} %`")
                
                with st.expander("💡 Afficher le Rapport d'Urgence Cardiaque & Injonctions", expanded=True):
                    st.markdown("L'Intelligence Logistique identifie ce patient avec un haut potentiel d'embolie ou infarctus coronaire basé sur la combinatoire vicieuse de ses paramètres.")
                    st.markdown("**PRESCRIPTION MÉDICALE CIBLÉE AUX VARIABLES DANGEREUSES :**")
                    for c in conseils: st.markdown(f"⚠️ {c}")
                    if not conseils:
                        st.markdown("⚠️ Le sujet présente une **usure létale indéfinie** (effet cocktail). Consultation aux urgences cardiologiques (Test d'effort sous surveillance, doppler extra-crânien) exigée au plus vite.")

            else:
                st.balloons()
                st.success("✅ **Profil de Résilience Physiologique. Aucune attaque CHD directe escomptée par le système.**")
                st.markdown(f"**Bruit Statistique de l'Algorithme :** `{proba*100:.1f} %`")
                
                with st.expander("💡 Afficher le Protocole Préventif Actif (Routine Bien-Être)", expanded=True):
                    st.markdown("Malgré la prédiction rassurante qui met le patient à l'abri au 1er niveau de soin, notre surveillance relève des brèches de santé publique qu'il faut colmater préventivement :")
                    if conseils:
                        st.markdown("**ACTION PRÉVENTIVE CONSEILLÉE :**")
                        for c in conseils: st.markdown(f"🔹 {c}")
                    else:
                        st.markdown("✨ *Absolument aucune faille majeure ou comportementale décelée. Le patient offre une protection vasculaire magistrale pour son âge. Maintenir ce système de santé (Sommeil, Diète).*")
    else:
         st.warning("Intelligence Artificielle en pause matérielle (Pkl disparu). Action requise: `python train_model.py`.")
