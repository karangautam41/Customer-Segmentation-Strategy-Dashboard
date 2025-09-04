import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
import os
import joblib
import logging
from typing import Tuple, Union, Optional

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- [FIX 1] Create an absolute path to the data file ---
# This ensures the script can find the CSV regardless of where you run it from.
script_dir = os.path.dirname(os.path.realpath(__file__))

#======================================================================
# SECTION 1: MACHINE LEARNING PIPELINE
#======================================================================

class PipelineConfig:
    """Configuration class for pipeline artifacts and paths."""
    # Use the script_dir variable to create a robust, absolute path
    DATA_FILE_PATH = os.path.join(script_dir, 'credit_card_data.csv')
    
    ARTIFACTS_DIR = 'artifacts'
    AUTOENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'autoencoder.h5')
    ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'encoder.h5')
    IMPUTER_PATH = os.path.join(ARTIFACTS_DIR, 'imputer.joblib')
    SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')
    PCA_PATH = os.path.join(ARTIFACTS_DIR, 'pca.joblib')
    KMEANS_PATH = os.path.join(ARTIFACTS_DIR, 'kmeans.joblib')

class SegmentationPipeline:
    """
    An end-to-end pipeline for segmenting credit card customers.
    """
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.reducer: Optional[Union[PCA, Model]] = None
        self.kmeans: Optional[KMeans] = None
        self.numeric_columns: list = []
        os.makedirs(self.config.ARTIFACTS_DIR, exist_ok=True)

    def _load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads data, cleans it, and preprocesses only the numeric columns.
        """
        logging.info(f"Loading data from local file: {self.config.DATA_FILE_PATH}...")
        try:
            df = pd.read_csv(self.config.DATA_FILE_PATH)
        except FileNotFoundError:
            logging.error(f"FATAL: The data file '{self.config.DATA_FILE_PATH}' was not found.")
            st.error(f"Error: The data file '{self.config.DATA_FILE_PATH}' was not found. Please ensure it's in the same directory as the app.")
            raise
        
        # --- [NEW] Data Cleaning: Drop non-numeric and identifier columns ---
        # These columns are categorical or IDs and are not suitable for this clustering model.
        cols_to_drop = [
            'CLIENTNUM', 'Attrition_Flag', 'Gender', 'Education_Level', 
            'Marital_Status', 'Income_Category', 'Card_Category',
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
        ]
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Identify only the remaining numeric columns for processing
        self.numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        logging.info(f"Identified {len(self.numeric_columns)} numeric columns for processing.")
        df_numeric = df[self.numeric_columns]

        logging.info("Imputing missing values...")
        self.imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(self.imputer.fit_transform(df_numeric), columns=self.numeric_columns)
        
        # --- [FIX 2] Updated Feature Engineering for the new dataset ---
        logging.info("Performing feature engineering tailored to the dataset...")
        
        # Use a small constant to avoid division by zero
        epsilon = 1e-6
        
        if 'Months_on_book' in df_imputed.columns and df_imputed['Months_on_book'].sum() > 0:
            df_imputed['MONTHLY_AVG_TRANSACTIONS'] = df_imputed['Total_Trans_Ct'] / (df_imputed['Months_on_book'] + epsilon)
            df_imputed['MONTHLY_AVG_TRANSACTION_AMT'] = df_imputed['Total_Trans_Amt'] / (df_imputed['Months_on_book'] + epsilon)
        
        if 'Total_Trans_Amt' in df_imputed.columns and 'Total_Revolving_Bal' in df_imputed.columns:
            df_imputed['TRANSACTION_TO_BALANCE_RATIO'] = df_imputed['Total_Trans_Amt'] / (df_imputed['Total_Revolving_Bal'] + epsilon)
        
        if 'Total_Trans_Amt' in df_imputed.columns and 'Total_Trans_Ct' in df_imputed.columns and df_imputed['Total_Trans_Ct'].sum() > 0:
            df_imputed['AVG_TRANSACTION_VALUE'] = df_imputed['Total_Trans_Amt'] / (df_imputed['Total_Trans_Ct'] + epsilon)
        
        # Replace any potential infinite values created during division
        df_imputed.replace([np.inf, -np.inf], 0, inplace=True)
        # Re-impute in case feature engineering created new NaNs
        df_imputed.fillna(df_imputed.mean(), inplace=True)

        logging.info("Scaling data...")
        self.scaler = StandardScaler()
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df_imputed), columns=df_imputed.columns)

        return df_imputed, df_scaled

    def _build_and_train_autoencoder(self, X_scaled: pd.DataFrame, encoding_dim: int = 3) -> Model:
        """Builds and trains a new autoencoder model."""
        logging.info("Building and training a new autoencoder...")
        input_dim = X_scaled.shape[1]

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(12, activation='relu')(input_layer)
        encoded = Dense(8, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu', name='encoded_layer')(encoded)
        decoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(12, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, shuffle=True, validation_split=0.1, verbose=0)

        encoder = Model(input_layer, encoded)

        autoencoder.save(self.config.AUTOENCODER_PATH)
        encoder.save(self.config.ENCODER_PATH)

        return encoder

    def run_pipeline(self, method: str = 'PCA', n_clusters: int = 4, n_components: int = 3, force_retrain: bool = False) -> Tuple[pd.DataFrame, float]:
        """
        Executes the full pipeline: load, preprocess, reduce dimensions, and cluster.
        """
        df_original, df_scaled = self._load_and_prepare_data()

        logging.info(f"Performing dimensionality reduction using {method}...")
        if method == 'PCA':
            if not force_retrain and os.path.exists(self.config.PCA_PATH):
                self.reducer = joblib.load(self.config.PCA_PATH)
                X_reduced = self.reducer.transform(df_scaled)
            else:
                self.reducer = PCA(n_components=n_components, random_state=42)
                X_reduced = self.reducer.fit_transform(df_scaled)

        elif method == 'Autoencoder':
            if not force_retrain and os.path.exists(self.config.ENCODER_PATH):
                self.reducer = load_model(self.config.ENCODER_PATH)
            else:
                self.reducer = self._build_and_train_autoencoder(df_scaled, n_components)
            X_reduced = self.reducer.predict(df_scaled)
        else:
            raise ValueError("Method must be 'PCA' or 'Autoencoder'")

        logging.info(f"Running K-Means clustering with {n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto', random_state=42)
        clusters = self.kmeans.fit_predict(X_reduced)
        df_original['cluster'] = clusters

        score = 0
        if len(set(clusters)) > 1:
            score = silhouette_score(X_reduced, clusters)
            logging.info(f"Clustering complete. Silhouette Score: {score:.4f}")

        self.save_artifacts()

        return df_original, score

    def save_artifacts(self):
        """Saves all fitted components of the pipeline."""
        logging.info("Saving pipeline artifacts...")
        if self.imputer: joblib.dump(self.imputer, self.config.IMPUTER_PATH)
        if self.scaler: joblib.dump(self.scaler, self.config.SCALER_PATH)
        if self.kmeans: joblib.dump(self.kmeans, self.config.KMEANS_PATH)

        if isinstance(self.reducer, PCA):
            joblib.dump(self.reducer, self.config.PCA_PATH)
        logging.info("Artifacts saved successfully.")

#======================================================================
# SECTION 2: STREAMLIT APPLICATION
#======================================================================

st.set_page_config(layout="wide", page_title="Customer Segmentation Dashboard")

# --- Page Title and Description ---
st.title("Interactive Customer Segmentation Dashboard 💳")
st.markdown("""
This dashboard allows you to explore customer segments from credit card data.
- **Choose your model:** Select a dimensionality reduction technique (PCA or Autoencoder).
- **Select cluster count:** Use the slider to define how many customer segments to create.
- **Visualize:** See the results in an interactive 3D plot.
- **Analyze:** View detailed profiles and actionable recommendations for each segment.
""")

# --- Data Preview Section ---
st.header("Data Preview and Schema")
with st.expander("Click to view the raw data and column descriptions"):
    try:
        preview_df = pd.read_csv(PipelineConfig.DATA_FILE_PATH)
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(preview_df.head())
        
        st.subheader("Dataset Columns")
        st.write(preview_df.columns.tolist())
        
    except FileNotFoundError:
        st.error(f"Error: The data file '{PipelineConfig.DATA_FILE_PATH}' was not found.")
    except Exception as e:
        st.error(f"An error occurred while loading the data preview: {e}")

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
reduction_method = st.sidebar.selectbox("1. Select Dimensionality Reduction Method", ('PCA', 'Autoencoder'))
num_clusters = st.sidebar.slider("2. Select Number of Clusters", min_value=2, max_value=10, value=4, step=1)
force_retrain = st.sidebar.checkbox("Force model retraining", value=False)

if force_retrain:
    st.sidebar.warning("Forcing a retrain will re-run the entire ML pipeline and may take a few minutes, especially for the Autoencoder.")

# --- Caching the pipeline run ---
@st.cache_data(show_spinner="Running the segmentation pipeline...")
def run_segmentation(method, n_clusters, retrain_trigger):
    """Function to run the ML pipeline."""
    pipeline = SegmentationPipeline()
    df_final, score = pipeline.run_pipeline(
        method=method,
        n_clusters=n_clusters,
        force_retrain=retrain_trigger
    )

    df_imputed = df_final.drop('cluster', axis=1)
    df_scaled = pd.DataFrame(pipeline.scaler.transform(df_imputed), columns=df_imputed.columns)

    if method == 'PCA':
        reduced_data = pipeline.reducer.transform(df_scaled)
    else: # Autoencoder
        reduced_data = pipeline.reducer.predict(df_scaled)

    if reduced_data.shape[1] < 3:
        pad_width = 3 - reduced_data.shape[1]
        reduced_data = np.pad(reduced_data, ((0, 0), (0, pad_width)), 'constant')

    df_plot = pd.DataFrame(reduced_data[:, :3], columns=['Component 1', 'Component 2', 'Component 3'])
    df_plot['cluster'] = df_final['cluster'].values

    return df_final, df_plot, score

# --- Main Dashboard Area ---
st.header("Segmentation Analysis")

if st.sidebar.button("Run Analysis"):
    try:
        df_final, df_plot, silhouette_score_val = run_segmentation(reduction_method, num_clusters, force_retrain)

        st.metric(label=f"Silhouette Score for {num_clusters} Clusters", value=f"{silhouette_score_val:.4f}")

        st.subheader("3D Cluster Visualization")
        fig = px.scatter_3d(
            df_plot,
            x='Component 1', y='Component 2', z='Component 3',
            color='cluster',
            title='Customer Segments in 3D Space',
            color_continuous_scale=px.colors.qualitative.Vivid,
            hover_name=df_plot.index
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=600, legend_title_text='Cluster')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Profiles & Recommendations")
        profile_df = df_final.groupby('cluster').mean()

        personas = {
            0: {"name": "Thrifty Spenders", "desc": "Tend to have lower balances and purchases. May use their card for smaller, regular transactions.", "reco": "Offer cashback rewards on everyday spending categories like groceries and gas. Introduce budget management tools."},
            1: {"name": "High-Value Transactors", "desc": "Characterized by high spending, high balances, and frequent payments. Likely use their card as a primary payment tool.", "reco": "Promote premium cards with travel benefits and higher credit limits. Offer exclusive access to events."},
            2: {"name": "Revolving Debt Users", "desc": "These customers maintain a high revolving balance relative to their credit limit and transactions. Purchase activity might be moderate.", "reco": "Offer low-interest balance transfer options or personal loans to consolidate debt. Provide financial literacy resources on managing credit."},
            3: {"name": "Conservative Users", "desc": "Have very low balances and minimal purchase activity. They own a card but use it infrequently.", "reco": "Run re-engagement campaigns with limited-time offers (e.g., 'Spend $50 and get $10 back'). Highlight security benefits of using a credit card."},
            4: {"name": "High-Frequency Shoppers", "desc": "Make a lot of transactions, but the individual value of each transaction is low. May have moderate balances.", "reco": "Promote a rewards program based on transaction counts rather than total spend. Offer 'Buy Now, Pay Later' (BNPL) options."},
        }

        for i in sorted(df_final['cluster'].unique()):
            persona = personas.get(i, {"name": f"Segment {i+1}", "desc": "A distinct group of customers with unique spending habits.", "reco": "Analyze spending patterns to develop tailored marketing campaigns."})
            
            with st.expander(f"**Cluster {i}: {persona['name']}**"):
                col1, col2 = st.columns([1.5, 2])
                with col1:
                    st.markdown(f"**Persona:** {persona['name']}")
                    st.write(persona['desc'])
                    st.markdown(f"**Recommendations:**")
                    st.info(persona['reco'])
                with col2:
                    st.dataframe(profile_df.loc[i].to_frame().style.highlight_max(axis=0))
    except FileNotFoundError:
        st.error("Execution stopped because the data file was not found.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e) 
else:
    st.info("Adjust the settings in the sidebar and click 'Run Analysis' to generate the customer segments.")