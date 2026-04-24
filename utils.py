# utils.py

def assign_persona(cluster):
    """
    Assign business-friendly persona based on cluster number
    """
    personas = {
        0: "💰 Budget Customer",
        1: "🙂 Standard Customer",
        2: "💎 Premium Customer",
        3: "🔥 High Spender",
        4: "🧠 Careful Saver"
    }
    return personas.get(cluster, "Unknown")


def get_cluster_summary(df):
    """
    Generate summary statistics for each cluster
    """
    summary = df.groupby("Cluster").mean(numeric_only=True)
    return summary
