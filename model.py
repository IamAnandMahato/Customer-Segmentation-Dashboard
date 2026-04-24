import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_train_model():
    df = pd.read_csv("Mall_Customers.csv")

    # Encode Gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # Features
    features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    return df, kmeans, scaler, le
