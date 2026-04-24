import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "Mall_Customers.csv")

    df = pd.read_csv(file_path)

    # Encode Gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # Features
    X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Model
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # ✅ RETURN ONLY 3 VALUES
    return df, kmeans, scaler
