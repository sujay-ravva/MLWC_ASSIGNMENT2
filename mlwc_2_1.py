
# Numerical arrays and mathematical operations
import numpy as np
# Dataframe handling and CSV input/output
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # safe for headless / script execution
# Plotting the Part (b) and Part (c) results
import matplotlib.pyplot as plt

# Creates the required train-test split
from sklearn.model_selection import train_test_split
# Standardises features before SVM training
from sklearn.preprocessing import StandardScaler
# RBF-kernel Support Vector Machine
from sklearn.svm import SVC
# Measures classification accuracy
from sklearn.metrics import accuracy_score


# =====================================================================
# 0. GLOBAL CONFIGURATION
# =====================================================================
N_SAMPLES        = 1000                          # samples per class per SNR
L                = 6                             # number of multipath taps
K_DB             = 9.0                           # Rician K-factor (dB)
MEAN_DELAY_LOS   = 30.0                          # LOS delay/decay constant (ns)
MEAN_DELAY_NLOS  = 100.0                         # NLOS delay/decay constant (ns)
SNR_DB_VALUES    = [0, 5, 10, 15, 20, 25, 30]     # SNR values to generate (dB)

DATASET_SEED     = 67                            # seed used for dataset generation
SPLIT_SEED       = 42                            # seed used for train/test splits

# Five features required by the assignment
FEATURES = ['kurtosis', 'skewness', 'rising_time', 'rms_delay_spread', 'k_factor']

# Map each SVM to its input feature set
CLASSIFIERS = {
    'SVM-1 (Kurtosis)':         ['kurtosis'],
    'SVM-2 (Skewness)':         ['skewness'],
    'SVM-3 (Rising time)':      ['rising_time'],
    'SVM-4 (RMS delay spread)': ['rms_delay_spread'],
    'SVM-5 (Rician K-factor)':  ['k_factor'],
    'SVM-6 (All 5 combined)':   FEATURES,
}

TRAIN_SNR = 25   # SNR used for the fixed-training experiment in Part (c)


# =====================================================================
# 1. DATASET GENERATION  (LOS = Rician, NLOS = Rayleigh channel taps)
#    -- identical logic/RNG-call-order to the provided generator, so the
#       output CSV is bit-for-bit the same given the same seed (67).
# =====================================================================
# Generate Rician channel realisations for the LOS class
def generate_LOS(n_samples):
    """
    Generate Rician (LOS) channel tap amplitudes and relative delays.

    - All L tap delays are drawn from Exp(MEAN_DELAY_LOS) and sorted;
      then normalised so the earliest arriving path has delay = 0 ns.
    - Tap 0 (earliest path) carries the dominant LOS component:
        amplitude = sqrt(K/(K+1)) * exp(j*phi),  phi ~ Uniform[0, 2*pi)
    - Taps 1..L-1 are scattered components sharing power 1/(K+1) with
      an exponential power-decay profile in delay.
    """
    # Convert K-factor from dB to linear scale
    K = 10 ** (K_DB / 10)
    channels = np.zeros((n_samples, L), dtype=complex)
    delays   = np.zeros((n_samples, L))

    for i in range(n_samples):
        # Draw random tap delays from an exponential distribution
        raw_delays = np.random.exponential(scale=MEAN_DELAY_LOS, size=L)
        raw_delays = np.sort(raw_delays)
        delays[i]  = raw_delays - raw_delays[0]

        # Random phase for the dominant LOS component
        phi = np.random.uniform(0, 2 * np.pi)
        channels[i, 0] = np.sqrt(K / (K + 1)) * np.exp(1j * phi)

        # Delays of the scattered LOS components
        scatter_delays = delays[i, 1:]
        decay_weights  = np.exp(-scatter_delays / MEAN_DELAY_LOS)
        scatter_powers = (decay_weights / decay_weights.sum()) / (K + 1)

        sigma = np.sqrt(scatter_powers / 2)
        channels[i, 1:] = (np.random.randn(L - 1) +
                            1j * np.random.randn(L - 1)) * sigma

    return channels, delays


# Generate Rayleigh channel realisations for the NLOS class
def generate_NLOS(n_samples):
    """
    Generate Rayleigh (NLOS) channel tap amplitudes and relative delays.
    All L taps are scattered (no dominant component, K = 0); tap powers
    follow an exponential decay profile with time constant MEAN_DELAY_NLOS.
    """
    channels = np.zeros((n_samples, L), dtype=complex)
    delays   = np.zeros((n_samples, L))

    for i in range(n_samples):
        raw_delays = np.random.exponential(scale=MEAN_DELAY_NLOS, size=L)
        raw_delays = np.sort(raw_delays)
        delays[i]  = raw_delays - raw_delays[0]

        # Apply exponential power decay across NLOS taps
        decay_weights = np.exp(-delays[i] / MEAN_DELAY_NLOS)
        tap_powers    = decay_weights / decay_weights.sum()

        sigma = np.sqrt(tap_powers / 2)
        channels[i] = (np.random.randn(L) +
                        1j * np.random.randn(L)) * sigma

    return channels, delays


# Add complex AWGN at the requested SNR
def add_noise(channels, snr_db):
    """Add complex AWGN so that total channel power / total noise power = SNR."""
    # Convert SNR from dB to linear scale
    snr_lin   = 10 ** (snr_db / 10)
    noise_var = 1.0 / (L * snr_lin)
    sigma_n   = np.sqrt(noise_var / 2)
    noise = (np.random.randn(*channels.shape) +
             1j * np.random.randn(*channels.shape)) * sigma_n
    return channels + noise


# Build the complete raw LOS/NLOS dataset
def build_dataset(filename='los_nlos_dataset.csv'):
    """Generate LOS/NLOS realisations at every SNR and save to CSV."""
    np.random.seed(DATASET_SEED)   # matches the provided generator exactly

    print("Generating LOS channels ...")
    los_ch, los_delays = generate_LOS(N_SAMPLES)

    print("Generating NLOS channels ...")
    nlos_ch, nlos_delays = generate_NLOS(N_SAMPLES)

    all_rows = []
    # Repeat the channel data at every required SNR
    for snr_db in SNR_DB_VALUES:
        print(f"  Adding noise at SNR = {snr_db} dB ...")
        los_noisy  = add_noise(los_ch,  snr_db)
        nlos_noisy = add_noise(nlos_ch, snr_db)

        for i in range(N_SAMPLES):
            row = {'snr_db': snr_db, 'label': 1}
            for tap in range(L):
                row[f'h_real_{tap}'] = los_noisy[i, tap].real
                row[f'h_imag_{tap}'] = los_noisy[i, tap].imag
                row[f'tau_{tap}']    = los_delays[i, tap]
            all_rows.append(row)

        for i in range(N_SAMPLES):
            row = {'snr_db': snr_db, 'label': -1}
            for tap in range(L):
                row[f'h_real_{tap}'] = nlos_noisy[i, tap].real
                row[f'h_imag_{tap}'] = nlos_noisy[i, tap].imag
                row[f'tau_{tap}']    = nlos_delays[i, tap]
            all_rows.append(row)

    # Convert generated rows into a dataframe
    df = pd.DataFrame(all_rows)
    df.to_csv(filename, index=False)
    print(f"Saved {filename}  ({len(df)} rows)\n")
    return df


# =====================================================================
# 2. PART (a): FEATURE EXTRACTION
#    Kurtosis, Skewness, Rising time, RMS delay spread, Rician K-factor
#    computed per-row from the L=6 tap amplitudes/delays (paper Eqs 2-9)
# =====================================================================
# Extract the five scalar features from each channel realisation
def extract_features(df, L=L, eps=1e-12):
    n = len(df)
    h_real = np.stack([df[f'h_real_{l}'].values for l in range(L)], axis=1)
    h_imag = np.stack([df[f'h_imag_{l}'].values for l in range(L)], axis=1)
    tau    = np.stack([df[f'tau_{l}'].values    for l in range(L)], axis=1)

    amp  = np.sqrt(h_real ** 2 + h_imag ** 2)      # |h_l|,  shape (n, L)
    amp2 = amp ** 2                                # tap power |h_l|^2

    mu    = amp.mean(axis=1)
    sigma = amp.std(axis=1)
    sigma_safe = np.where(sigma < eps, eps, sigma)

    # 1. Kurtosis (Eq. 2)
    m4 = ((amp - mu[:, None]) ** 4).mean(axis=1)
    kurtosis = m4 / (sigma_safe ** 4)

    # 2. Skewness (Eq. 5)
    m3 = ((amp - mu[:, None]) ** 3).mean(axis=1)
    skewness = m3 / (sigma_safe ** 3)

    # 3. Rising time (Eq. 6): delay of strongest tap - min delay
    # Find the strongest tap in each realisation
    strongest_idx = np.argmax(amp, axis=1)
    tau_at_peak   = tau[np.arange(n), strongest_idx]
    tau_min       = tau.min(axis=1)
    rising_time   = tau_at_peak - tau_min

    # 4. RMS delay spread (Eq. 7-8)
    # Total received power across all taps
    power_sum = amp2.sum(axis=1)
    power_sum_safe = np.where(power_sum < eps, eps, power_sum)
    tau_mean_excess = (tau * amp2).sum(axis=1) / power_sum_safe
    rms_delay_spread = np.sqrt(
        (((tau - tau_mean_excess[:, None]) ** 2) * amp2).sum(axis=1) / power_sum_safe
    )

    # 5. Rician K-factor
    #    Standard power-ratio estimator: ratio of the strongest tap's power
    #    to the total power of the remaining taps, in dB. This is the same
    #    definition the dataset generator itself uses (see its
    #    sanity_check() function) to confirm the synthetic data hits the
    #    target K_DB = 9.0 for LOS / ~0 dB for NLOS, and it is far more
    #    robust with only L=6 taps than the paper's literal small-sample
    #    amplitude-variance form of Eq. (9), which folds the dominant tap
    #    itself into the variance term and is therefore very noise-sensitive.
    peak_power  = amp2.max(axis=1)
    total_power = amp2.sum(axis=1)
    k_factor = 10 * np.log10(peak_power / (total_power - peak_power + eps))

    df = df.copy()
    df['kurtosis']         = kurtosis
    df['skewness']         = skewness
    df['rising_time']      = rising_time
    df['rms_delay_spread'] = rms_delay_spread
    df['k_factor']         = k_factor
    return df


# =====================================================================
# 3. PART (b): SINGLE-FEATURE vs COMBINED-FEATURE SVMs (per SNR)
# =====================================================================
# Train and test one SVM using data from one SNR
def train_test_at_snr(df_snr, feature_cols, random_state=SPLIT_SEED):
    """80/20 stratified split -> StandardScaler (fit on train only) ->
    RBF SVM (C=1). Returns accuracy, fitted scaler, fitted model, split."""
    X = df_snr[feature_cols].values
    y = df_snr['label'].values

    # Use the required stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Fit scaling only on the training data
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train the required RBF SVM with C = 1
    model = SVC(kernel='rbf', C=1)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred) * 100.0
    return acc, scaler, model, (X_train, X_test, y_train, y_test)


# Run all six SVMs separately at every SNR
def run_part_b(df):
    snr_values = sorted(df['snr_db'].unique())
    results = {name: [] for name in CLASSIFIERS}
    svm6_bundle = {}   # snr -> {'scaler', 'model', 'split'} for SVM-6, reused in Part (c)

    # Evaluate the classifiers independently at each SNR
    for snr in snr_values:
        df_snr = df[df['snr_db'] == snr].reset_index(drop=True)
        for name, feats in CLASSIFIERS.items():
            acc, scaler, model, split = train_test_at_snr(df_snr, feats)
            results[name].append(acc)
            if name == 'SVM-6 (All 5 combined)':
                svm6_bundle[snr] = {'scaler': scaler, 'model': model, 'split': split}

    # ---- Table ----
    # Store Part (b) accuracies in a table
    results_df = pd.DataFrame(results, index=snr_values)
    results_df.index.name = 'SNR (dB)'
    print("=" * 70)
    print("PART (b): Classification Accuracy (%) vs SNR")
    print("=" * 70)
    print(results_df.round(2).to_string())
    results_df.round(2).to_csv('part_b_accuracy_table.csv')
    print("\nSaved part_b_accuracy_table.csv")

    # ---- Plot ----
    # Create the Part (b) accuracy plot
    plt.figure(figsize=(9, 6))
    markers = ['o', 's', '^', 'D', 'v', '*']
    for (name, accs), m in zip(results.items(), markers):
        lw = 3 if 'combined' in name else 1.6
        ms = 9 if 'combined' in name else 6
        plt.plot(snr_values, accs, marker=m, linewidth=lw, markersize=ms, label=name)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Accuracy (%)')
    plt.title('LOS/NLOS SVM Classification Accuracy vs SNR\n'
              '(RBF kernel, C = 1, 80/20 train-test split per SNR)')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim(45, 102)
    plt.tight_layout()
    plt.savefig('accuracy_vs_snr.png', dpi=150)
    plt.close()
    print("Saved accuracy_vs_snr.png\n")

    return results_df, svm6_bundle, snr_values


# =====================================================================
# 4. PART (c): FIXED-25dB-TRAINED SVM-6 vs MATCHED-SNR SVM-6
# =====================================================================
# Compare matched-SNR training with fixed 25 dB training
def run_part_c(svm6_bundle, snr_values):
    # ---- Curve 1: "Train = Test SNR" (re-uses Part (b) SVM-6 results) ----
    # Accuracy when SVM-6 is trained at the same SNR as the test data
    matched_acc = []
    for snr in snr_values:
        scaler = svm6_bundle[snr]['scaler']
        model  = svm6_bundle[snr]['model']
        _, X_test, _, y_test = svm6_bundle[snr]['split']
        y_pred = model.predict(scaler.transform(X_test))
        matched_acc.append(accuracy_score(y_test, y_pred) * 100.0)

    # ---- Fixed model: SVM-6 trained ONCE on SNR = 25 dB training data ----
    fixed_scaler = svm6_bundle[TRAIN_SNR]['scaler']
    fixed_model  = svm6_bundle[TRAIN_SNR]['model']

    # ---- Curve 2: "Train at 25 dB" evaluated on every SNR's test split ----
    # Accuracy when the SVM-6 model remains fixed at 25 dB
    fixed_acc = []
    for snr in snr_values:
        _, X_test, _, y_test = svm6_bundle[snr]['split']
        X_test_scaled = fixed_scaler.transform(X_test)   # 25 dB scaler reused
        y_pred = fixed_model.predict(X_test_scaled)
        fixed_acc.append(accuracy_score(y_test, y_pred) * 100.0)

    # ---- Table ----
    # Store both Part (c) curves and their accuracy gap
    table = pd.DataFrame({
        'Train = Test SNR (%)':  matched_acc,
        'Train at 25 dB (%)':    fixed_acc,
        'Gap (matched - fixed)': np.array(matched_acc) - np.array(fixed_acc)
    }, index=snr_values)
    table.index.name = 'Test SNR (dB)'
    print("=" * 70)
    print("PART (c): Fixed-SNR training vs matched-SNR training (SVM-6)")
    print("=" * 70)
    print(table.round(2).to_string())
    table.round(2).to_csv('part_c_accuracy_table.csv')
    print("\nSaved part_c_accuracy_table.csv")

    # ---- Plot ----
    # Create the Part (c) generalisation plot
    plt.figure(figsize=(8, 6))
    plt.plot(snr_values, matched_acc, marker='o', linewidth=2.5, markersize=8,
              label='Train = Test SNR (SVM-6, retrained at each SNR)')
    plt.plot(snr_values, fixed_acc, marker='s', linewidth=2.5, markersize=8,
              label='Train at 25 dB (SVM-6, fixed model)')
    plt.axvline(TRAIN_SNR, color='gray', linestyle='--', alpha=0.6,
                label='Training SNR = 25 dB')

    plt.xlabel('Test SNR (dB)')
    plt.ylabel('Classification Accuracy (%)')
    plt.title('Generalisation of a Fixed-SNR-Trained SVM-6\n'
              'vs a Model Retrained at Every SNR')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim(45, 102)
    plt.tight_layout()
    plt.savefig('train25_vs_matched.png', dpi=150)
    plt.close()
    print("Saved train25_vs_matched.png\n")

    return table


# =====================================================================
# MAIN
# =====================================================================
# Run the complete assignment workflow when this file is executed
if __name__ == "__main__":

    # ---- Step 1: generate the raw dataset ----
    # Step 1: generate the raw channel dataset
    df_raw = build_dataset('los_nlos_dataset.csv')

    # ---- Step 2: Part (a) feature extraction ----
    # Step 2: calculate the five required features
    df_feat = extract_features(df_raw, L=L)
    df_feat.to_csv('los_nlos_features.csv', index=False)
    print("PART (a): Saved los_nlos_features.csv "
          f"({len(df_feat)} rows, features = {FEATURES})\n")

    print("Mean feature values by class @ SNR = 30 dB:")
    print(df_feat[df_feat.snr_db == 30].groupby('label')[FEATURES].mean()
          .rename(index={1: 'LOS (+1)', -1: 'NLOS (-1)'}).round(3), "\n")

    print("Mean feature values by class @ SNR = 0 dB:")
    print(df_feat[df_feat.snr_db == 0].groupby('label')[FEATURES].mean()
          .rename(index={1: 'LOS (+1)', -1: 'NLOS (-1)'}).round(3), "\n")

    # ---- Step 3: Part (b) ----
    # Step 3: run the six SVM experiments
    results_df, svm6_bundle, snr_values = run_part_b(df_feat)

    # ---- Step 4: Part (c) ----
    # Step 4: run the fixed-SNR generalisation experiment
    part_c_table = run_part_c(svm6_bundle, snr_values)

    print("=" * 70)
    print("DONE. Files generated in the current directory:")
    print("  los_nlos_dataset.csv")
    print("  los_nlos_features.csv")
    print("  accuracy_vs_snr.png")
    print("  part_b_accuracy_table.csv")
    print("  train25_vs_matched.png")
    print("  part_c_accuracy_table.csv")
    print("=" * 70)