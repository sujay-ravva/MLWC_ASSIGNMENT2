import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# =====================================================================
# 0. GLOBAL CONFIGURATION
# =====================================================================

N_SAMPLES_PER_POINT = 200          # Samples generated for each 16-QAM point
SNR_DB_VALUES = [0, 5, 10, 15, 20, 25, 30]
DATASET_SEED = 67
KMEANS_SEED = 42

K_VALUES = range(2, 21)
K_FINAL = 16
TRAIN_SNR = 25


# =====================================================================
# 1. DATASET GENERATION
# =====================================================================

def generate_dataset(filename='qam16_dataset.csv'):

    # 16-QAM constellation coordinates specified in the assignment
    levels = np.array([-3, -1, 1, 3])

    # Create all 16 combinations of I and Q coordinates
    constellation = np.array(
        [(i, q) for i in levels for q in levels],
        dtype=float
    )

    # Normalize the constellation so that average symbol energy is 1
    avg_energy = np.mean(
        constellation[:, 0]**2 + constellation[:, 1]**2
    )

    constellation = constellation / np.sqrt(avg_energy)

    np.random.seed(DATASET_SEED)

    all_rows = []

    for snr_db in SNR_DB_VALUES:

        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)

        # AWGN standard deviation for unit average symbol energy
        noise_sigma = np.sqrt(1 / (2 * snr_linear))

        for symbol_id, (i, q) in enumerate(constellation):

            # Generate the required number of received samples
            noise_i = np.random.randn(N_SAMPLES_PER_POINT) * noise_sigma
            noise_q = np.random.randn(N_SAMPLES_PER_POINT) * noise_sigma

            rx_i = i + noise_i
            rx_q = q + noise_q

            for n in range(N_SAMPLES_PER_POINT):

                all_rows.append({
                    'snr_db': snr_db,
                    'symbol_id': symbol_id,
                    'Rx': rx_i[n],
                    'Ry': rx_q[n]
                })

    df = pd.DataFrame(all_rows)

    df.to_csv(filename, index=False)

    print(f"Saved {filename} ({len(df)} rows)")
    return df


# =====================================================================
# 2. PART (a): FEATURE EXTRACTION
# =====================================================================

def extract_features(df, filename='qam16_features.csv'):

    df = df.copy()

    # Instantaneous amplitude
    df['r'] = np.sqrt(df['Rx']**2 + df['Ry']**2)

    # Instantaneous phase in radians
    df['theta'] = np.arctan(df['Ry'] / df['Rx'])

    df.to_csv(filename, index=False)

    print(f"Saved {filename} ({len(df)} rows)")

    return df


# =====================================================================
# 3. CLUSTER PURITY
# =====================================================================

def cluster_purity(y_true, cluster_labels):

    total_correct = 0

    # Examine each predicted cluster separately
    for cluster in np.unique(cluster_labels):

        cluster_true_labels = y_true[cluster_labels == cluster]

        if len(cluster_true_labels) == 0:
            continue

        # Assign the cluster to its most common true symbol
        counts = np.bincount(cluster_true_labels.astype(int),
                             minlength=16)

        total_correct += np.max(counts)

    return total_correct / len(y_true) * 100.0


# =====================================================================
# 4. PART (b): K-MEANS EVALUATION
# =====================================================================

def run_part_b(df):

    # Use only the required 25 dB data
    df_25 = df[df['snr_db'] == 25].reset_index(drop=True)

    y_true = df_25['symbol_id'].values

    # -------------------------------------------------------------
    # Feature sets specified in the assignment
    # -------------------------------------------------------------

    feature_sets = {
        'Feature Set 1 (Rx, Ry)': ['Rx', 'Ry'],
        'Feature Set 2 (r, theta)': ['r', 'theta'],
        'Feature Set 3 (Rx, Ry, r, theta)': ['Rx', 'Ry', 'r', 'theta']
    }

    # -------------------------------------------------------------
    # Evaluate K = 2 to 20 using Cartesian coordinates
    # -------------------------------------------------------------

    X_cartesian = df_25[['Rx', 'Ry']].values

    inertia_values = []
    silhouette_values = []

    for k in K_VALUES:

        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=KMEANS_SEED
        )

        labels = kmeans.fit_predict(X_cartesian)

        inertia_values.append(kmeans.inertia_)

        # Silhouette score requires at least 2 clusters
        silhouette_values.append(
            silhouette_score(X_cartesian, labels)
        )

    # -------------------------------------------------------------
    # Plot Inertia and Silhouette side-by-side
    # -------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        list(K_VALUES),
        inertia_values,
        marker='o'
    )

    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Within-Cluster Sum of Squares (Inertia)')
    axes[0].set_title('Inertia vs. K')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        list(K_VALUES),
        silhouette_values,
        marker='o'
    )

    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Coefficient')
    axes[1].set_title('Silhouette Coefficient vs. K')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('part_b_inertia_silhouette.png', dpi=150)
    plt.close()

    print("Saved part_b_inertia_silhouette.png")

    # -------------------------------------------------------------
    # K = 16 clustering using Cartesian coordinates
    # -------------------------------------------------------------

    kmeans_16 = KMeans(
        n_clusters=K_FINAL,
        n_init=10,
        random_state=KMEANS_SEED
    )

    cluster_labels = kmeans_16.fit_predict(X_cartesian)

    centroids = kmeans_16.cluster_centers_

    # -------------------------------------------------------------
    # Plot the 16 clusters and their centroids
    # -------------------------------------------------------------

    plt.figure(figsize=(8, 7))

    plt.scatter(
        df_25['Rx'],
        df_25['Ry'],
        c=cluster_labels,
        s=12,
        alpha=0.45,
        cmap='tab20'
    )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='X',
        s=180,
        c='black',
        edgecolors='white',
        linewidths=1.5,
        label='Cluster Centroids'
    )

    plt.xlabel('$R_x$')
    plt.ylabel('$R_y$')
    plt.title('16-QAM K-means Clustering at SNR = 25 dB')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()

    plt.savefig('part_b_k16_constellation.png', dpi=150)
    plt.close()

    print("Saved part_b_k16_constellation.png")

    # -------------------------------------------------------------
    # Cluster purity for all three feature sets
    # -------------------------------------------------------------

    purity_results = {}

    for name, features in feature_sets.items():

        X = df_25[features].values

        # StandardScaler is applied to Feature Set 3 as required
        if name == 'Feature Set 3 (Rx, Ry, r, theta)':

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        kmeans = KMeans(
            n_clusters=K_FINAL,
            n_init=10,
            random_state=KMEANS_SEED
        )

        labels = kmeans.fit_predict(X)

        purity = cluster_purity(y_true, labels)

        purity_results[name] = purity

    purity_df = pd.DataFrame(
        {
            'Feature Set': list(purity_results.keys()),
            'Cluster Purity (%)': list(purity_results.values())
        }
    )

    purity_df.to_csv(
        'part_b_cluster_purity.csv',
        index=False
    )

    print("\nPart (b): Cluster Purity")
    print("=" * 60)
    print(purity_df.round(2).to_string(index=False))
    print("\nSaved part_b_cluster_purity.csv")

    # -------------------------------------------------------------
    # Save K evaluation results
    # -------------------------------------------------------------

    k_results = pd.DataFrame({
        'K': list(K_VALUES),
        'Inertia': inertia_values,
        'Silhouette': silhouette_values
    })

    k_results.to_csv(
        'part_b_k_evaluation.csv',
        index=False
    )

    print("Saved part_b_k_evaluation.csv")

    return (
        purity_df,
        kmeans_16,
        centroids,
        k_results
    )


# =====================================================================
# 5. PART (c): ADAPTIVE vs FIXED TEMPLATE
# =====================================================================

def run_part_c(df):

    y_all = df['symbol_id'].values

    adaptive_purity = []
    fixed_purity = []

    # -------------------------------------------------------------
    # Strategy 2: train the fixed template only at 25 dB
    # -------------------------------------------------------------

    df_train = df[df['snr_db'] == TRAIN_SNR]

    X_train = df_train[['Rx', 'Ry']].values
    y_train = df_train['symbol_id'].values

    fixed_kmeans = KMeans(
        n_clusters=K_FINAL,
        n_init=10,
        random_state=KMEANS_SEED
    )

    fixed_kmeans.fit(X_train)

    fixed_centroids = fixed_kmeans.cluster_centers_

    # -------------------------------------------------------------
    # Evaluate both strategies at every SNR
    # -------------------------------------------------------------

    for snr in SNR_DB_VALUES:

        df_snr = df[df['snr_db'] == snr]

        X_test = df_snr[['Rx', 'Ry']].values
        y_test = df_snr['symbol_id'].values

        # Strategy 1: retrain K-means at the current SNR
        adaptive_kmeans = KMeans(
            n_clusters=K_FINAL,
            n_init=10,
            random_state=KMEANS_SEED
        )

        adaptive_labels = adaptive_kmeans.fit_predict(X_test)

        adaptive_purity.append(
            cluster_purity(y_test, adaptive_labels)
        )

        # Strategy 2: use the fixed 25 dB centroids
        distances = np.linalg.norm(
            X_test[:, None, :] -
            fixed_centroids[None, :, :],
            axis=2
        )

        fixed_labels = np.argmin(distances, axis=1)

        fixed_purity.append(
            cluster_purity(y_test, fixed_labels)
        )

    # -------------------------------------------------------------
    # Save Part (c) results
    # -------------------------------------------------------------

    part_c_df = pd.DataFrame({
        'SNR (dB)': SNR_DB_VALUES,
        'Adaptive K-means Purity (%)': adaptive_purity,
        'Fixed Template Purity (%)': fixed_purity
    })

    part_c_df.to_csv(
        'part_c_cluster_purity.csv',
        index=False
    )

    print("\nPart (c): Cluster Purity vs SNR")
    print("=" * 70)
    print(part_c_df.round(2).to_string(index=False))
    print("\nSaved part_c_cluster_purity.csv")

    # -------------------------------------------------------------
    # Plot both strategies
    # -------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        SNR_DB_VALUES,
        adaptive_purity,
        marker='o',
        linewidth=2.5,
        markersize=7,
        label='Adaptive K-means'
    )

    plt.plot(
        SNR_DB_VALUES,
        fixed_purity,
        marker='s',
        linewidth=2.5,
        markersize=7,
        label='Fixed Template (trained at 25 dB)'
    )

    plt.axvline(
        TRAIN_SNR,
        linestyle='--',
        alpha=0.6,
        label='Training SNR = 25 dB'
    )

    plt.xlabel('SNR (dB)')
    plt.ylabel('Cluster Purity (%)')
    plt.title(
        'Adaptive K-means vs. Fixed 25-dB Template'
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)

    plt.tight_layout()

    plt.savefig(
        'part_c_adaptive_vs_fixed.png',
        dpi=150
    )

    plt.close()

    print("Saved part_c_adaptive_vs_fixed.png")

    return part_c_df


# =====================================================================
# 6. MAIN
# =====================================================================

if __name__ == '__main__':

    print("=" * 70)
    print("QUESTION 2 - 16-QAM K-MEANS CLUSTERING")
    print("=" * 70)

    # -------------------------------------------------------------
    # Part (a): Generate dataset and extract features
    # -------------------------------------------------------------

    df_raw = generate_dataset(
        'qam16_dataset.csv'
    )

    df_features = extract_features(
        df_raw,
        'qam16_features.csv'
    )

    # -------------------------------------------------------------
    # Part (b): K-means analysis
    # -------------------------------------------------------------

    purity_df, kmeans_16, centroids, k_results = run_part_b(
        df_features
    )

    # -------------------------------------------------------------
    # Part (c): Adaptive vs fixed-template clustering
    # -------------------------------------------------------------

    part_c_df = run_part_c(
        df_features
    )

    # -------------------------------------------------------------
    # Final output summary
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nFiles generated:")
    print("1. qam16_dataset.csv")
    print("2. qam16_features.csv")
    print("3. part_b_inertia_silhouette.png")
    print("4. part_b_k16_constellation.png")
    print("5. part_b_cluster_purity.csv")
    print("6. part_b_k_evaluation.csv")
    print("7. part_c_cluster_purity.csv")
    print("8. part_c_adaptive_vs_fixed.png")

    print("=" * 70)