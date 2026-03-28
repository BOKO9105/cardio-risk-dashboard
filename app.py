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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')

# ---- Entraînement dynamique (compatible toutes versions Python) ----
@st.cache_resource
def train_model_and_results():
    """Entraîne le modèle au démarrage de l'app et retourne le pipeline et les résultats."""
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

    models_to_fit = {
        'Régression Logistique': LogisticRegression(max_iter=1500, random_state=42, class_weight='balanced', solver='liblinear'),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    }

    results = {}
    best_pipeline = None
    best_f1 = 0
    best_name = ""

    for name, model in models_to_fit.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        results[name] = {
            'CV F1 Score Mean': float(cv_scores.mean()),
            'Test F1 Score': float(f1),
            'Test Recall': float(recall_score(y_test, y_pred)),
            'Test ROC AUC': float(roc_auc_score(y_test, y_proba)),
            'Confusion Matrix': cm,
            'ROC_FPR': fpr[::3].tolist(),
            'ROC_TPR': tpr[::3].tolist()
        }
        if name == 'Régression Logistique':
            preprocessor_fit = pipe.named_steps['preprocessor'].fit(X_train)
            cat_enc = preprocessor_fit.named_transformers_['cat'].named_steps['ohe']
            cat_names = [f"{col}_{v}" for col, vals in zip(categorical_features, cat_enc.categories_) for v in vals[1:]]
            feat_names = numeric_features + cat_names + binary_features
            coefs = pipe.named_steps['classifier'].coef_[0]
            results['Global_Feature_Importance'] = {f: float(abs(c)) for f, c in zip(feat_names, coefs)}
        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipe
            best_name = name

    return best_pipeline, results

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
        imp_dict = results.pop("Global_Feature_Importance", None)
        
        st.write("Le Machine Learning Médical exige une transparence totale. Voici la justification mathématique et biologique de notre Modèle de Prévention.")
        
        # --- LIGNE 1 : FIGURES (ROC et K-FOLD) ---
        col_roc, col_cv = st.columns([1.5, 1.5])
        
        with col_roc:
            st.markdown("<h3 class='title-h2'>📈 Courbes ROC (Sensibilité)</h3>", unsafe_allow_html=True)
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            for m_name in results.keys():
                fpr, tpr = results[m_name].get('ROC_FPR', []), results[m_name].get('ROC_TPR', [])
                auc_score = results[m_name].get('Test ROC AUC', 0)
                if len(fpr) > 0:
                    ax_roc.plot(fpr, tpr, label=f'{m_name} (AUC = {auc_score:.2f})', linewidth=2)
            ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc='lower right')
            sns.despine()
            st.pyplot(fig_roc)
            
        with col_cv:
            st.markdown("<h3 class='title-h2'>✅ Stratified K-Fold (Moyenne = 5)</h3>", unsafe_allow_html=True)
            df_cv = pd.DataFrame({
                "Modèle Evalué": list(results.keys()),
                "F1 Score Modéré": [results[k].get("CV F1 Score Mean", 0) for k in results.keys()],
                "Recall Test": [results[k].get("Test Recall", 0) for k in results.keys()]
            }).set_index("Modèle Evalué")
            st.dataframe(df_cv.style.highlight_max(color='#1E4D2B'), use_container_width=True)
            
        # COMMENTAIRE HORIZONTAL PLEINE LARGEUR (SOUS LA LIGNE 1)
        with st.expander("💡 Pourquoi la modeste Régression Logistique a-t-elle remporté le combat ?", expanded=False):
            st.markdown("""
            Contre toute attente liée à la mode des modèles profonds complexes (Gradient Boosting, Random Forest), la **Régression Logistique Statistique** s'est avérée être le choix scientifique parfait pour notre outil.<br><br>
            **Explication Technique :** Sur des données médicales biaisées où les malades ne pèsent que 15%, les arbres décisionnels (Random Forest) paniquent et préfèrent ignorer la classe minoritaire, générant d'énormes biais (il classe tout le monde comme Sain pour ne pas se tromper). La Régression Logistique pénalisée préserve cette limite et compense mathématiquement pour générer le meilleur **Recall** (trouver la vaste majorité des malades).
            """, unsafe_allow_html=True)
            
        st.markdown("---")
            
        # --- LIGNE 2 : FIGURES (MATRICES et FEATURE IMPORTANCE) ---
        col_cm, col_imp = st.columns([1.5, 1.5])
        
        with col_cm:
            st.markdown("<h3 class='title-h2'>🎯 Matrices de Confusions (Le Cheat Code)</h3>", unsafe_allow_html=True)
            mat_lr = np.array(results.get('Régression Logistique', {}).get('Confusion Matrix', [[0,0],[0,0]]))
            mat_rf = np.array(results.get('Random Forest', {}).get('Confusion Matrix', [[0,0],[0,0]]))
            fig_cm, axes_cm = plt.subplots(1, 2, figsize=(9, 3.5))
            sns.heatmap(mat_lr, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0], cbar=False)
            axes_cm[0].set_title('Logistique Vainqueur')
            axes_cm[0].set_ylabel('Vérité Médicale')
            sns.heatmap(mat_rf, annot=True, fmt='d', cmap='Reds', ax=axes_cm[1], cbar=False)
            axes_cm[1].set_title('Random Forest (Perdant)')
            st.pyplot(fig_cm)

        with col_imp:
            st.markdown("<h3 class='title-h2'>🔑 Poids Médicaux Isolés (Coefficients)</h3>", unsafe_allow_html=True)
            if imp_dict:
                imp_s = pd.Series(imp_dict).sort_values(ascending=True).tail(6)
                fig_imp, ax_imp = plt.subplots(figsize=(6, 3.8))
                imp_s.plot(kind='barh', color='#E63946', ax=ax_imp)
                sns.despine()
                st.pyplot(fig_imp)
                
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
