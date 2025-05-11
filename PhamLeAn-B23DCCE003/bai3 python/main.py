import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
RESULTS_CSV_PATH = 'results.csv'
PLAYER_ID_COLUMN = 'Player' # Adjust if your player name column is different
OUTPUT_DIR = 'clustering_pca_results'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# IMPORTANT: Verify and adjust these column names to match your 'results.csv'
# These are the numeric statistics that will be used for clustering.
FEATURE_COLUMNS_FOR_CLUSTERING = [
    'Age', 'MP', 'Starts', 'Min',
    'Gls', 'Ast', 'CrdY', 'CrdR',
    'xG', 'xAG',
    'PrgC', 'PrgP', 'PrgR',
    # Per 90 stats - common FBRef names are Gls.1, Ast.1, etc.
    # If your CSV uses Gls/90, xG/90, these must be changed here.
    'Gls.1', 'Ast.1', 'xG.1', 'xAG.1',
    # Goalkeeping - common FBRef names
    'GA90', 'Save%', 'CS%',
    'PKsv%',  # Penalty Kick Save %. FBRef might use Save%.1 under PKs section for GKs.
              # Or it could be a custom name from your scraper.
    # Shooting - common FBRef names
    'SoT%', 'SoT/90', 'G/Sh', 'Dist',
    # Passing - common FBRef names
    'Cmp', 'Cmp%', 'PrgDist', # For Total Cmp, Cmp%, Prog Pass Dist
    # For Cmp% for Short, Medium, Long. FBRef shows them as "Cmp% (Short)".
    # Your scraper might create "Cmp%_S", "Cmp% (Short)", "PassCompletionPercentageShort", etc.
    'Cmp%_S', 'Cmp%_M', 'Cmp%_L' # Example names, ensure these are correct
]

# --- 1. Load and Prepare Data ---
def load_and_preprocess_for_clustering(csv_path, feature_cols_config, player_id_col):
    """Loads, cleans, and prepares data for clustering."""
    print(f"Loading data from '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return None, None, None

    # Ensure player ID column exists
    if player_id_col not in df.columns:
        print(f"Error: Player ID column '{player_id_col}' not found in the CSV.")
        # Attempt to find common alternatives if player_id_col is the default 'Player'
        if player_id_col == 'Player':
            found_alt = False
            for alt_col in ['Player Name', 'Name', 'Nome']: # Add other common alternatives
                if alt_col in df.columns:
                    player_id_col = alt_col
                    print(f"Using alternative player ID column: '{player_id_col}'")
                    found_alt = True
                    break
            if not found_alt:
                return None, None, None
        else: # If a custom player_id_col was specified and not found
             return None, None, None


    df_original_for_analysis = df.copy() # Keep original for later analysis if needed

    # Select actual feature columns present in the DataFrame
    actual_feature_cols = [col for col in feature_cols_config if col in df.columns]
    missing_cols = [col for col in feature_cols_config if col not in df.columns]

    if not actual_feature_cols:
        print("Error: None of the specified feature columns for clustering were found in the CSV.")
        print("Please check the 'FEATURE_COLUMNS_FOR_CLUSTERING' list in the script.")
        return None, None, None
    if missing_cols:
        print("\nWarning: The following configured feature columns were NOT found in the CSV and will be skipped:")
        for col in missing_cols:
            print(f"  - {col}")
        print("Consider updating 'FEATURE_COLUMNS_FOR_CLUSTERING'.\n")

    print(f"Using the following {len(actual_feature_cols)} columns for clustering: {actual_feature_cols}")
    features_df = df[actual_feature_cols].copy()

    # Convert "N/a" and other string NaNs to np.nan, then to numeric
    for col in features_df.columns:
        if features_df[col].dtype == 'object':
            features_df[col] = features_df[col].replace(['N/a', 'NaN', 'nan'], np.nan)
            # Handle percentage strings if not already done by previous steps
            if features_df[col].str.contains('%', na=False).any():
                 features_df[col] = features_df[col].str.rstrip('%').astype('float') / 100.0
            else:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        else: # If already numeric, ensure it's float for consistency with imputation
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')


    # Impute missing values (e.g., with median)
    # Note: For inapplicable stats (GK stats for outfielders), 0 might be better.
    # Here, using median as a general approach.
    print("Imputing missing values using column medians...")
    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform(features_df)
    features_imputed_df = pd.DataFrame(features_imputed, columns=actual_feature_cols, index=df.index)

    # Scale features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=actual_feature_cols, index=df.index)

    player_ids = df[player_id_col]
    return features_scaled_df, player_ids, df_original_for_analysis, actual_feature_cols


# --- 2. Determine Optimal K (Elbow and Silhouette) ---
def find_optimal_k(scaled_data, max_k=15, output_path_prefix="optimal_k"):
    """Calculates and plots Elbow method and Silhouette scores."""
    print("\n--- Determining Optimal K ---")
    k_range = range(2, max_k + 1)
    inertia_values = []
    silhouette_values = []

    print("Calculating Inertia (Elbow Method)...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', verbose=0)
        kmeans.fit(scaled_data)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertia_values, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Errors)")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(list(k_range))
    plt.grid(True)
    elbow_plot_path = os.path.join(OUTPUT_DIR, f"{output_path_prefix}_elbow_plot.png")
    plt.savefig(elbow_plot_path)
    plt.show()
    print(f"Elbow method plot saved to: {elbow_plot_path}")

    print("Calculating Silhouette Scores...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', verbose=0)
        cluster_labels_temp = kmeans.fit_predict(scaled_data)
        if len(np.unique(cluster_labels_temp)) > 1: # Silhouette score is only defined if there is more than 1 cluster
            silhouette_avg = silhouette_score(scaled_data, cluster_labels_temp)
            silhouette_values.append(silhouette_avg)
        else:
            silhouette_values.append(-1) # Or some indicator of invalid score for 1 cluster if k=1 was allowed
            print(f"Warning: Only 1 cluster found for k={k}. Silhouette score cannot be computed meaningfully.")


    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouette_values, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Silhouette Analysis for Optimal k")
    plt.xticks(list(k_range))
    plt.grid(True)
    silhouette_plot_path = os.path.join(OUTPUT_DIR, f"{output_path_prefix}_silhouette_plot.png")
    plt.savefig(silhouette_plot_path)
    plt.show()
    print(f"Silhouette analysis plot saved to: {silhouette_plot_path}")
    print("Please analyze these plots to choose an optimal 'k'.")
    print("Elbow: Look for a point where the rate of decrease in inertia sharply slows.")
    print("Silhouette: Look for a peak in the silhouette score.")


# --- 3. K-means Clustering ---
def apply_kmeans(scaled_data, n_clusters):
    """Applies K-means algorithm."""
    print(f"\n--- Applying K-means with k={n_clusters} ---")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', verbose=0)
    cluster_labels = kmeans.fit_predict(scaled_data)
    print(f"K-means clustering complete. Players assigned to {n_clusters} clusters.")
    return cluster_labels


# --- 4. PCA and Visualization ---
def pca_and_plot_clusters(scaled_data, cluster_labels, output_path_prefix="pca_clusters"):
    """Performs PCA and plots 2D clusters."""
    print("\n--- PCA and Plotting Clusters ---")
    pca = PCA(n_components=2, random_state=42)
    pca_results_2d = pca.fit_transform(scaled_data)
    print(f"PCA complete. Explained variance by 2 components: {pca.explained_variance_ratio_.sum():.2%}")

    pca_df = pd.DataFrame(data=pca_results_2d, columns=['PC1', 'PC2'], index=scaled_data.index)
    pca_df['Cluster'] = cluster_labels

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=50, alpha=0.7)
    plt.title(f'Player Clusters (PCA Reduced to 2D) - {len(np.unique(cluster_labels))} Clusters')
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend(title='Cluster')
    plt.grid(True)
    pca_plot_path = os.path.join(OUTPUT_DIR, f"{output_path_prefix}_2d_plot.png")
    plt.savefig(pca_plot_path)
    plt.show()
    print(f"PCA cluster plot saved to: {pca_plot_path}")
    return pca_df


# --- 5. Analyze Clusters (Cluster Profiling) ---
def analyze_clusters(original_df, used_feature_cols, cluster_labels, player_id_col, output_path_prefix="cluster_profiles"):
    """Analyzes and profiles the generated clusters."""
    print("\n--- Cluster Analysis & Comments ---")
    # Make sure to use the original data before scaling for easier interpretation of means
    # but use the features that were actually part of clustering
    analysis_df = original_df.copy()
    analysis_df['Cluster'] = cluster_labels

    # For profiling, use the imputed (but not scaled) version of features if available,
    # or re-select and convert original features. For simplicity, using original_df and selected cols.
    # Need to handle N/a to numeric for calculating means if not already done on original_df for these cols.
    numeric_profile_df = original_df[used_feature_cols].copy()
    for col in numeric_profile_df.columns:
        if numeric_profile_df[col].dtype == 'object':
            numeric_profile_df[col] = numeric_profile_df[col].replace(['N/a', 'NaN', 'nan'], np.nan)
            if numeric_profile_df[col].str.contains('%', na=False).any():
                 numeric_profile_df[col] = numeric_profile_df[col].str.rstrip('%').astype('float') / 100.0
            else:
                numeric_profile_df[col] = pd.to_numeric(numeric_profile_df[col], errors='coerce')
        else:
            numeric_profile_df[col] = pd.to_numeric(numeric_profile_df[col], errors='coerce')

    numeric_profile_df['Cluster'] = cluster_labels
    cluster_profiles = numeric_profile_df.groupby('Cluster')[used_feature_cols].mean()

    print("\nCluster Profiles (Mean values of features for each cluster):")
    print(cluster_profiles)
    profiles_path = os.path.join(OUTPUT_DIR, f"{output_path_prefix}.csv")
    cluster_profiles.to_csv(profiles_path)
    print(f"Cluster profiles saved to: {profiles_path}")

    print("\nDistribution of Player Positions (Pos) within each cluster (if 'Pos' column exists):")
    if 'Pos' in original_df.columns:
        position_distribution = analysis_df.groupby('Cluster')['Pos'].value_counts(normalize=False).unstack(fill_value=0)
        print(position_distribution)
        pos_dist_path = os.path.join(OUTPUT_DIR, f"{output_path_prefix}_position_distribution.csv")
        position_distribution.to_csv(pos_dist_path)
        print(f"Position distribution saved to: {pos_dist_path}")
    else:
        print("'Pos' column not found in original data for position distribution analysis.")

    print("\nTo provide comments on the results (as required by the assignment):")
    print("1. Justify your choice of 'k' (number of clusters) using the Elbow and Silhouette plots.")
    print("2. For each cluster, examine its profile (mean statistics).")
    print("   - What are the defining characteristics? (e.g., high 'Gls' and 'xG' for attackers,")
    print("     high 'Save%' for GKs, high 'PrgP' and 'Tkl' for certain midfielders/defenders).")
    print("3. Observe the 2D PCA plot. Do the clusters appear distinct? Is there overlap?")
    print("4. If 'Pos' data is available, how do player positions distribute across your clusters?")
    print("   This can help validate or understand the nature of the statistically derived groups.")


# --- Main Execution ---
def main():
    """Main function to run the clustering and PCA analysis."""
    print("Starting 'Bài 3': K-means Clustering and PCA Analysis.")

    # 1. Load and preprocess data
    processed_features_df, player_ids, df_original, actual_feature_cols_used = \
        load_and_preprocess_for_clustering(RESULTS_CSV_PATH, FEATURE_COLUMNS_FOR_CLUSTERING, PLAYER_ID_COLUMN)

    if processed_features_df is None or processed_features_df.empty:
        print("\nExiting due to data loading or preprocessing errors.")
        return
    if not actual_feature_cols_used: # Double check
        print("\nNo features were selected or available for clustering. Exiting.")
        return

    # 2. Determine optimal K
    find_optimal_k(processed_features_df, max_k=12) # Adjust max_k if needed

    # --- User Input for Chosen K ---
    chosen_k = -1
    while chosen_k < 2 :
        try:
            k_input = input(f"\nBased on the plots, enter the chosen number of clusters (k >= 2, e.g., 3, 4, or 5): ")
            chosen_k = int(k_input)
            if chosen_k < 2:
                print("Number of clusters must be at least 2.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # 3. Apply K-means
    cluster_labels = apply_kmeans(processed_features_df, chosen_k)

    # 4. PCA and Visualization
    pca_df_with_clusters = pca_and_plot_clusters(processed_features_df, cluster_labels)

    # 5. Analyze Clusters
    analyze_clusters(df_original, actual_feature_cols_used, cluster_labels, PLAYER_ID_COLUMN)

    print(f"\nAnalysis complete. All outputs saved in '{OUTPUT_DIR}' directory.")
    print("Remember to include your justification for 'k' and comments on cluster results in your report.")

if __name__ == '__main__':
    main()