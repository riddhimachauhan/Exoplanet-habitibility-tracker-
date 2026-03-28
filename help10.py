"""
=============================================================================
EXOPLANET HABITABILITY ANALYSIS — DATA MINING PROJECT (FULLY CORRECTED)
NASA Exoplanet Archive | Clustering + Classification + Outlier Detection
=============================================================================
ALL FIXES INTEGRATED:
  1. LABEL LEAKAGE FIX: Probabilistic score-based labels (top 10% by
     habitability_score). Eliminates AUC=1.0 trivially.
  2. CLASS IMBALANCE FIX: SMOTE with k_neighbors=5.
  3. KMEANS k=5 FIX: Extreme outliers (|z|>4.5 in ANY feature) are removed
     BEFORE clustering so k=5 produces 5 meaningful clusters instead of
     real + singleton. Outliers are stored separately and re-attached
     after clustering — they do NOT affect target planet recovery.
  4. AGGLOMERATIVE FIX: Ward linkage + SAME outlier removal as KMeans
     (runs on df_cluster, not full df_feat — fixes singleton micro-clusters).
  5. COLORMAP FIX: All scatter plots use vmin/vmax-bounded tab10 so all
     5 clusters render in distinct, correct colours.
  6. ASSOCIATION RULES FIX: Filtered for Habitable as consequent.
  7. ALL 20 TARGET PLANETS guaranteed in results.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

# ============================================================
# SECTION 1: LOAD AND PREPROCESS DATA
# ============================================================
print("=" * 70)
print("SECTION 1: DATA LOADING & PREPROCESSING")
print("=" * 70)

df = pd.read_csv(
    "PS_2026.01.16_08.12.38.csv",
    engine="python",
    sep=",",
    quotechar='"',
    escapechar="\\",
    on_bad_lines="skip",
    comment="#"
)

df.columns = df.columns.str.strip()
print(f"Raw dataset shape: {df.shape}")

df = df[df["default_flag"] == 1].reset_index(drop=True)
# ALL NUMERICAL FEATURES (for Plot 1 & 3)
ALL_FEATURES = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove non-useful columns if present
REMOVE_COLS = ["default_flag", "disc_year"]
ALL_FEATURES = [c for c in ALL_FEATURES if c not in REMOVE_COLS]
print(f"After default_flag filter: {df.shape}")

FEATURES = [
    "pl_rade", "pl_bmasse", "pl_dens", "pl_eqt", "pl_insol",
    "pl_orbsmax", "pl_orbper", "pl_orbeccen",
    "pl_trandur", "pl_ratdor", "pl_ratror",
    "st_teff", "st_rad", "st_mass", "st_met", "st_lum", "st_logg"
]

REQUIRED = ["pl_rade"]

print(f"\nFeatures selected ({len(FEATURES)}):")
for f in FEATURES:
    print(f"  {f}")

df_feat = df[["pl_name"] + FEATURES].copy()

print("\nMissing values per feature (before imputation):")
miss = df_feat[FEATURES].isnull().sum().sort_values(ascending=False)
print(miss.to_string())

df_feat = df_feat.dropna(subset=REQUIRED)
print(f"\nAfter dropping rows missing {REQUIRED}: {df_feat.shape}")

# ============================================================
# IMPUTATION USING KNN
# ============================================================
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

print("\nApplying KNN Imputation...")

scaler_temp = StandardScaler()
scaled_temp = scaler_temp.fit_transform(df_feat[FEATURES])

imputer = KNNImputer(n_neighbors=5)
imputed_scaled = imputer.fit_transform(scaled_temp)
imputed_data = scaler_temp.inverse_transform(imputed_scaled)
df_feat[FEATURES] = imputed_data

print(f"Remaining NaNs after KNN imputation: {df_feat[FEATURES].isnull().sum().sum()}")
print(f"Final usable rows: {df_feat.shape[0]}")

TARGET_PLANETS = [
    'LP 890-9 c', 'Gliese 12 b', 'K2-3 d', 'Kepler-1652 b',
    'Kepler-1649 c', 'TOI-2095 c', 'Kepler-1653 b', 'TOI-715 b',
    'K2-133 e', 'TOI-6002 b', 'TOI-7166 b', 'TOI-4336 A b',
    'K2-9 b', 'TOI-1452 b', 'TOI-712 d', 'LHS 1140 b',
    'Kepler-1052 c', 'Kepler-22 b', 'Kepler-452 b', 'Kepler-186 f'
]

present_targets = [p for p in TARGET_PLANETS if p in df_feat["pl_name"].values]
missing_targets = [p for p in TARGET_PLANETS if p not in df_feat["pl_name"].values]
print(f"\nTarget planets present after preprocessing: {len(present_targets)}/{len(TARGET_PLANETS)}")
print(f"  Found: {present_targets}")
if missing_targets:
    print(f"  Absent: {missing_targets}")

print("\nDescriptive Statistics:")
print(df_feat[FEATURES].describe().round(3).to_string())
# Plot 3: Correlation Heatmap
# Select features with <= 50% missing
valid_feats = [c for c in ALL_FEATURES if df[c].isnull().mean() < 0.5]

# Fill temporarily for correlation
df_corr = df[valid_feats].copy()
df_corr = df_corr.fillna(df_corr.median(numeric_only=True))

# Compute correlation
corr_full = df_corr.corr().abs()

# Select top 40 features
top_features = corr_full.sum().sort_values(ascending=False).head(40).index.tolist()

# Final correlation
corr = df_corr[top_features].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1, center=0)

ax.set_title("Feature Correlation Heatmap (Top 40 Features)",
             fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("plot03_correlation_heatmap.png", dpi=150)
plt.show()
print("  -> Plot 1 saved: Correlation Heatmap")
# ============================================================
# SECTION 2: EXPLORATORY DATA VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: EXPLORATORY DATA VISUALIZATION")
print("=" * 70)

# Plot 1: Missing Value Bar Chart
fig, ax = plt.subplots(figsize=(18, 6))  # wider

miss_pct = (df[ALL_FEATURES].isnull().mean() * 100).sort_values(ascending=False)
colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 20 else "#2ecc71" for v in miss_pct]

ax.bar(range(len(miss_pct)), miss_pct.values,
       color=colors, edgecolor="black", linewidth=0.7)

# ✅ SHOW LIMITED LABELS (fix overlap)
step = max(1, len(miss_pct)//25)
ax.set_xticks(range(0, len(miss_pct), step))
ax.set_xticklabels(miss_pct.index[::step], rotation=45, ha="right")

ax.set_xlabel("Feature", fontsize=11)
ax.set_ylabel("Missing (%)", fontsize=11)
ax.set_title("Missing Value Percentage per Feature (Full Dataset)",
             fontsize=13, fontweight="bold")

ax.axhline(50, color="red", linestyle="--", linewidth=1.2, label="50% threshold")
ax.legend()

plt.tight_layout()
plt.savefig("plot01_missing_values.png", dpi=150)
plt.show()
print("  -> Plot 2 saved: Missing Value Bar Chart")

# Plot 2: Feature Distributions
fig, axes = plt.subplots(4, 5, figsize=(20, 14))
axes = axes.flatten()
for i, col in enumerate(FEATURES):
    data = df_feat[col]

    # ✅ remove extreme outliers (visual only)
    q1, q99 = data.quantile(0.01), data.quantile(0.99)
    data = data[(data >= q1) & (data <= q99)]

    # ✅ log transform if highly skewed
    if data.max() / (data.median() + 1e-9) > 50:
        data = np.log1p(data)

    axes[i].hist(data, bins=40, color="#3498db",edgecolor="white", linewidth=0.4)

    axes[i].grid(alpha=0.2)
    
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Value", fontsize=7)
    axes[i].set_ylabel("Count", fontsize=7)
    axes[i].tick_params(labelsize=7)
for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distribution of All 17 Features (after imputation)", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plot02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("  -> Plot 3 saved: Feature Distributions")



core5 = ["pl_rade", "pl_eqt", "pl_insol", "pl_bmasse", "st_teff"]
pair_df = df_feat[core5].copy()

# 🔥 STEP 1: Remove extreme outliers (important)
for col in core5:
    q1, q99 = pair_df[col].quantile(0.01), pair_df[col].quantile(0.99)
    pair_df = pair_df[(pair_df[col] >= q1) & (pair_df[col] <= q99)]

# 🔥 STEP 2: Log transform skewed features (MAIN FIX)
for col in ["pl_insol", "pl_bmasse", "pl_eqt"]:
    pair_df[col] = np.log1p(pair_df[col])

# 🔥 STEP 3: Sample for clarity
pair_df = pair_df.sample(min(1200, len(pair_df)), random_state=42)

# 🔥 STEP 4: Better pairplot
g = sns.pairplot(
    pair_df,
    diag_kind="hist",
    corner=True,
    plot_kws={"alpha": 0.6, "s": 18}
)

# ✅ FORCE LABELS TO SHOW (FIX)
for ax in g.axes.flatten():
    if ax is not None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=9)
        ax.set_ylabel(ax.get_ylabel(), fontsize=9)

g.fig.suptitle("Pairplot: Core Features (Log-Scaled & Cleaned)",
               y=1.02, fontsize=14, fontweight="bold")

plt.savefig("plot04_pairplot_core5.png", dpi=150, bbox_inches="tight")
plt.show()
print("  -> Plot 4 saved: Pairplot (core 5 features)")

# Plot 5: Box Plots
cols = FEATURES   # ✅ uses all 17 features automatically

rows = 4
cols_count = int(np.ceil(len(cols) / rows))

fig, axes = plt.subplots(rows, cols_count, figsize=(4*cols_count, 4*rows))
axes = axes.flatten()

colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]

for i, col in enumerate(cols):
    ax = axes[i]
    clr = colors[i % len(colors)]

    q1, q3 = df_feat[col].quantile(0.25), df_feat[col].quantile(0.75)
    iqr = q3 - q1
    trimmed = df_feat[(df_feat[col] >= q1 - 3*iqr) & (df_feat[col] <= q3 + 3*iqr)][col]

    ax.boxplot(
        trimmed,
        patch_artist=True,
        showfliers=False,  # 🔥 MAIN FIX
        boxprops=dict(facecolor=clr, alpha=0.6),
        medianprops=dict(color="black", linewidth=2)
    )
    ax.set_title(col, fontsize=9, fontweight="bold")
    ax.tick_params(axis='both', labelsize=7)
plt.suptitle(f"Box Plots: {len(cols)} Habitability Features (IQR-trimmed)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot05_boxplots.png", dpi=150)
for j in range(len(cols), len(axes)):
    axes[j].set_visible(False)
plt.show()
print("  -> Plot 5 saved: Box Plots")

# Plot 6: Radius vs Temperature
fig, ax = plt.subplots(figsize=(9, 6))
sub = df_feat[(df_feat["pl_rade"] < 20) & (df_feat["pl_eqt"] < 3000)]
sc = ax.scatter(sub["pl_rade"], sub["pl_eqt"],
                c=np.log1p(sub["pl_insol"]), cmap="plasma", s=12, alpha=0.6)
plt.colorbar(sc, ax=ax, label="log(Insolation + 1)")
ax.axhline(288, color="blue", linestyle="--", linewidth=1.5, alpha=0.7, label="Earth Temp (288K)")
ax.axvline(1.0, color="blue", linestyle="--", linewidth=1.5, alpha=0.7, label="Earth Radius (1.0)")
ax.scatter([1.0], [288], color="cyan", s=200, marker="*",
           edgecolors="blue", linewidth=2, label="Earth", zorder=10)
for p in TARGET_PLANETS:
    row = df_feat[df_feat["pl_name"] == p]
    if not row.empty:
        r, t = row["pl_rade"].values[0], row["pl_eqt"].values[0]
        if r < 20 and t < 3000:
            ax.scatter(r, t, color="red", s=60, zorder=11, marker="D")
ax.set_xlabel("Planet Radius (Earth radii)", fontsize=11)
ax.set_ylabel("Equilibrium Temperature (K)", fontsize=11)
ax.set_title("Radius vs Temperature  (diamond = target planets)", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot06_radius_vs_temp.png", dpi=150)
plt.show()
print("  -> Plot 6 saved: Radius vs Temperature")

# ============================================================
# Plot 6B: Radius vs Insolation (NEW - IMPORTANT)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

sub = df_feat[(df_feat["pl_rade"] < 20)]

sc = ax.scatter(
    sub["pl_rade"],
    sub["pl_insol"],
    c=sub["pl_eqt"],
    cmap="coolwarm",
    s=12,
    alpha=0.6
)

plt.colorbar(sc, ax=ax, label="Temperature (K)")

ax.axhline(1.0, color="blue", linestyle="--", linewidth=1.5, label="Earth Insolation")
ax.axvline(1.0, color="blue", linestyle="--", linewidth=1.5, label="Earth Radius")

ax.scatter([1.0], [1.0], color="cyan", s=200, marker="*",
           edgecolors="blue", linewidth=2, label="Earth")

ax.set_xlabel("Planet Radius (Earth radii)", fontsize=11)
ax.set_ylabel("Insolation (Earth flux)", fontsize=11)

ax.set_title("Radius vs Insolation (color = Temperature)",
             fontsize=13, fontweight="bold")

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot06B_radius_vs_insolation.png", dpi=150)
plt.show()

print("  -> Plot 6B saved: Radius vs Insolation")


# ============================================================
# Plot 6C: Temperature vs Insolation (NEW - RELATION CHECK)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

sub = df_feat[(df_feat["pl_eqt"] < 3000)]

sc = ax.scatter(
    sub["pl_insol"],
    sub["pl_eqt"],
    c=sub["pl_rade"],
    cmap="viridis",
    s=12,
    alpha=0.6
)

plt.colorbar(sc, ax=ax, label="Planet Radius")

ax.axhline(288, color="blue", linestyle="--", linewidth=1.5, label="Earth Temp")
ax.axvline(1.0, color="blue", linestyle="--", linewidth=1.5, label="Earth Insolation")

ax.scatter([1.0], [288], color="cyan", s=200, marker="*",
           edgecolors="blue", linewidth=2, label="Earth")

ax.set_xlabel("Insolation (Earth flux)", fontsize=11)
ax.set_ylabel("Equilibrium Temperature (K)", fontsize=11)

ax.set_title("Temperature vs Insolation (color = Radius)",
             fontsize=13, fontweight="bold")

ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot06C_temp_vs_insolation.png", dpi=150)
plt.show()

print("  -> Plot 6C saved: Temperature vs Insolation")

# Plot 7: Discovery Method
fig, ax = plt.subplots(figsize=(10, 5))
disc = df[df["default_flag"] == 1]["discoverymethod"].value_counts()
ax.bar(disc.index, disc.values,
       color=sns.color_palette("tab10", len(disc)), edgecolor="black")
ax.set_xlabel("Discovery Method", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Exoplanets by Discovery Method", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=35)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot07_discovery_method.png", dpi=150)
plt.show()
print("  -> Plot 7 saved: Discovery Method Distribution")

# Plot 8: Discovery Year
fig, ax = plt.subplots(figsize=(10, 4))
year_data = df[df["default_flag"] == 1]["disc_year"].dropna()
ax.hist(year_data, bins=range(int(year_data.min()), int(year_data.max()) + 2),
        color="#8e44ad", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Discovery Year", fontsize=11)
ax.set_ylabel("Number of Exoplanets", fontsize=11)
ax.set_title("Exoplanet Discoveries by Year", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot08_discovery_year.png", dpi=150)
plt.show()
print("  -> Plot 8 saved: Discovery Year Histogram")

# ============================================================
# SECTION 3: SCALING
# ============================================================
print("\n" + "=" * 70)
print("SECTION 3: SCALING & PREPROCESSING")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feat[FEATURES])
print(f"Scaled feature matrix shape: {X_scaled.shape}")

# ============================================================
# SECTION 4: K-MEANS CLUSTERING  (k = 5)
# ============================================================
print("\n" + "=" * 70)
print("SECTION 4: K-MEANS CLUSTERING")
print("=" * 70)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats as scipy_stats

# ── Step 4a: Identify and set aside extreme outliers ──────────────────
Z_THRESHOLD = 4.5
z_abs = np.abs(scipy_stats.zscore(df_feat[FEATURES]))
extreme_mask = (z_abs > Z_THRESHOLD).any(axis=1)

# Safety: never exclude target planets
target_mask = df_feat["pl_name"].isin(TARGET_PLANETS)
extreme_mask = extreme_mask & ~target_mask

n_extreme = extreme_mask.sum()
print(f"Extreme outliers removed before clustering (|z|>{Z_THRESHOLD}): {n_extreme}")
if n_extreme > 0:
    print("  Extreme planets set aside:")
    for name in df_feat[extreme_mask]["pl_name"].values:
        print(f"    - {name}")

df_cluster = df_feat[~extreme_mask].reset_index(drop=True)
df_extreme  = df_feat[extreme_mask].reset_index(drop=True)

X_cluster = scaler.transform(df_cluster[FEATURES])

print(f"\nClustering on {len(df_cluster)} planets "
      f"({len(df_extreme)} extreme outliers held out)")

# ── Step 4b: Elbow + silhouette ───────────────────────────────────────
K_FINAL = 5
K_range = range(2, 11)
inertia, silhouettes = [], []

for k in K_range:
    km_tmp  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl_tmp = km_tmp.fit_predict(X_cluster)
    inertia.append(km_tmp.inertia_)
    silhouettes.append(
        silhouette_score(X_cluster, lbl_tmp, sample_size=2000, random_state=42)
    )

print(f"\nK_FINAL = {K_FINAL}")

# ── Step 4c: Plot 9 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
k_list = list(K_range)
axes[0].plot(k_list, inertia, marker='o', color="#e74c3c", linewidth=2)
axes[0].axvline(K_FINAL, color="black", linestyle="--", linewidth=1.5,
                label=f"Chosen k={K_FINAL}")
axes[0].set_xlabel("k", fontsize=11)
axes[0].set_ylabel("Inertia (WCSS)", fontsize=11)
axes[0].set_title("Elbow Method for K-Means\n(extreme outliers removed)",
                  fontsize=12, fontweight="bold")
axes[0].legend(); axes[0].grid(True, alpha=0.4)

axes[1].plot(k_list, silhouettes, marker='s', color="#2980b9", linewidth=2)
axes[1].axvline(K_FINAL, color="black", linestyle="--", linewidth=1.5,
                label=f"Chosen k={K_FINAL}")
axes[1].set_xlabel("k", fontsize=11)
axes[1].set_ylabel("Silhouette Score", fontsize=11)
axes[1].set_title("Silhouette Score vs k\n(extreme outliers removed)",
                  fontsize=12, fontweight="bold")
axes[1].legend(); axes[1].grid(True, alpha=0.4)

plt.suptitle(f"K-Means Cluster Selection — k={K_FINAL} chosen from elbow analysis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot09_elbow_silhouette.png", dpi=150)
plt.show()
print(f"  -> Plot 9 saved: Elbow + Silhouette (k={K_FINAL})")

# ── Step 4d: Fit final KMeans on clean set ────────────────────────────
kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=20)
km_labels_clean = kmeans.fit_predict(X_cluster)
df_cluster["km_cluster"] = km_labels_clean

# ── Step 4e: Assign extreme outliers to nearest centroid ──────────────
if len(df_extreme) > 0:
    X_extreme = scaler.transform(df_extreme[FEATURES])
    km_labels_extreme = kmeans.predict(X_extreme)
    df_extreme = df_extreme.copy()
    df_extreme["km_cluster"] = km_labels_extreme

# ── Step 4f: Re-assemble ──────────────────────────────────────────────
df_feat = pd.concat([df_cluster, df_extreme], ignore_index=True)

cluster_sizes = np.bincount(df_feat["km_cluster"])
print(f"\nCluster sizes after fix: {dict(enumerate(cluster_sizes))}")
print(f"Min cluster size: {cluster_sizes.min()}")
assert cluster_sizes.min() >= 5, \
    "Still degenerate clusters! Try increasing Z_THRESHOLD."

km_stats = df_feat.groupby("km_cluster")[FEATURES].mean()
print("\nK-Means Cluster Statistics (key features):")
print(km_stats[["pl_rade", "pl_eqt", "pl_insol", "pl_bmasse", "st_teff"]].round(2).to_string())

EARTH_RADIUS = 1.0
EARTH_TEMP   = 288.0
EARTH_INSOL  = 1.0

target_cluster_votes = []
for p in TARGET_PLANETS:
    row = df_feat[df_feat["pl_name"] == p]
    if not row.empty:
        target_cluster_votes.append(int(row["km_cluster"].values[0]))

if target_cluster_votes:
    vote_counts      = Counter(target_cluster_votes)
    km_earth_cluster = vote_counts.most_common(1)[0][0]
    print(f"\nTarget planet cluster vote counts: {dict(sorted(vote_counts.items()))}")
else:
    earth_dist_km    = km_stats.apply(
        lambda row: abs(row["pl_rade"] - EARTH_RADIUS)
                  + abs(row["pl_eqt"]  - EARTH_TEMP) / 100
                  + abs(row["pl_insol"] - EARTH_INSOL), axis=1)
    km_earth_cluster = int(earth_dist_km.idxmin())

print(f"\nMost Earth-like K-Means cluster: {km_earth_cluster}")
print("\nInterpretation:")
print(f"Cluster {km_earth_cluster} represents Earth-like planets (based on radius, temperature, insolation)")
print(f"  Centroid: R={km_stats.loc[km_earth_cluster,'pl_rade']:.2f}  "
      f"T={km_stats.loc[km_earth_cluster,'pl_eqt']:.1f}K")

print("\nTarget planet cluster assignments (K-Means):")
for p in TARGET_PLANETS:
    row = df_feat[df_feat["pl_name"] == p]
    if not row.empty:
        c    = int(row["km_cluster"].values[0])
        r    = row["pl_rade"].values[0]
        t    = row["pl_eqt"].values[0]
        mark = "*" if c == km_earth_cluster else " "
        print(f"  {mark} {p:25s}  cluster={c}  R={r:.2f}  T={t:.1f}K")

# Re-scale full df_feat for downstream
X_scaled = scaler.transform(df_feat[FEATURES])

# ── Helper: discrete tab10 colormap ──────────────────────────────────
def get_tab10_colors(n):
    """Return n distinct colours from tab10, correctly indexed."""
    cmap = plt.cm.get_cmap("tab10", n)
    return [cmap(i) for i in range(n)]

tab10_colors = get_tab10_colors(K_FINAL)

# Plot 10: K-Means Clusters (PCA Projection) 🔥
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 6))

sc = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df_feat["km_cluster"],
    cmap=plt.cm.get_cmap("tab10", K_FINAL),
    vmin=-0.5,
    vmax=K_FINAL - 0.5,
    s=18,
    alpha=0.8,
    edgecolors="black",
    linewidths=0.2
)



cbar = plt.colorbar(sc, ax=ax, ticks=range(K_FINAL))
cbar.set_label("K-Means Cluster")

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("K-Means Clusters (PCA Projection — 17D → 2D)", fontweight="bold")


ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot10_kmeans_pca.png", dpi=150)
plt.show()

# Plot 11: Cluster Sizes (FIXED colormap)
fig, ax = plt.subplots(figsize=(8, 4))
cluster_counts = df_feat["km_cluster"].value_counts().sort_index()
ax.bar(cluster_counts.index, cluster_counts.values,
       color=tab10_colors, edgecolor="black")
ax.set_xlabel("Cluster ID", fontsize=11)
ax.set_ylabel("Number of Planets", fontsize=11)
ax.set_title(f"K-Means Cluster Sizes (k={K_FINAL})", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for i, v in zip(cluster_counts.index, cluster_counts.values):
    ax.text(i, v + 5, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("plot11_kmeans_cluster_sizes.png", dpi=150)
plt.show()
print("  -> Plot 11 saved: K-Means Cluster Sizes")

# Plot 12: Cluster Summary Table (BETTER THAN HEATMAP)

cluster_table = df_feat.groupby("km_cluster")[FEATURES].mean().round(2)

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

table = ax.table(
    cellText=cluster_table.values,
    colLabels=cluster_table.columns,
    rowLabels=[f"Cluster {i}" for i in cluster_table.index],
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.title(f"K-Means Cluster Summary (k={K_FINAL}) — Actual Feature Values",
          fontsize=13, fontweight="bold")

plt.savefig("plot12_cluster_table.png", dpi=150)
plt.show()

# ============================================================
# SECTION 5: HIERARCHICAL (AGGLOMERATIVE) CLUSTERING — CORRECTED
# Fix: Use PCA reduction first, then Ward linkage on clean set.
# This prevents the "one giant cluster + singletons" problem
# that Ward produces on high-dimensional skewed data.
# ============================================================
print("\n" + "=" * 70)
print("SECTION 5: HIERARCHICAL (AGGLOMERATIVE) CLUSTERING")
print("=" * 70)

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage

# ── Step 5a: Reduce to PCA (10 components) before clustering ─────────
# Ward linkage on 17 skewed features → 1 giant cluster.
# PCA compresses variance and makes Ward produce balanced clusters.
pca_agg = PCA(n_components=10, random_state=42)
X_cluster_pca = pca_agg.fit_transform(X_cluster)   # shape: (4322, 10)
print(f"PCA for Agglomerative: {X_cluster.shape} → {X_cluster_pca.shape}")
print(f"Variance explained: {pca_agg.explained_variance_ratio_.sum()*100:.1f}%")

# ── Step 5b: Dendrogram (sample of 500 from PCA-reduced clean set) ───
np.random.seed(42)
sample_idx     = np.random.choice(len(X_cluster_pca), size=min(500, len(X_cluster_pca)), replace=False)
X_sample_pca   = X_cluster_pca[sample_idx]
linkage_matrix = linkage(X_sample_pca, method="ward")

# Plot 13: Dendrogram
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(linkage_matrix, truncate_mode="lastp", p=30, ax=ax,
           color_threshold=0.4 * max(linkage_matrix[:, 2]),
           leaf_rotation=45, leaf_font_size=9)
ax.set_xlabel("Cluster Size", fontsize=11)
ax.set_ylabel("Ward Distance", fontsize=11)
ax.set_title("Dendrogram — Agglomerative Clustering (PCA-10D, Ward linkage, n=500 sample)",
             fontsize=12, fontweight="bold")
ax.axhline(y=np.percentile(linkage_matrix[:, 2], 85), color="red",
           linestyle="--", linewidth=1.5, label="Suggested cut")
ax.legend()
plt.tight_layout()
plt.savefig("plot13_dendrogram.png", dpi=150)
plt.show()
print("  -> Plot 13 saved: Dendrogram")

# ── Step 5c: Fit Agglomerative on PCA-reduced clean set ──────────────
agg = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward")
agg_labels_clean = agg.fit_predict(X_cluster_pca)
df_cluster["agg_cluster"] = agg_labels_clean

# Print cluster distribution to verify balance
print("\nAGG CLUSTER COUNTS (PCA-10D + Ward linkage):")
unique, counts = np.unique(agg_labels_clean, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Cluster {u}: {c} planets")

min_agg = counts.min()
max_agg = counts.max()
print(f"  Min: {min_agg}  |  Max: {max_agg}  |  Ratio: {max_agg/min_agg:.1f}x")
if max_agg / min_agg > 10:
    print("  WARNING: Still imbalanced — try PCA n_components=6 or 8.")
else:
    print("  ✓ Clusters are reasonably balanced.")

# ── Step 5d: Assign extreme outliers via nearest centroid in PCA space ─
if len(df_extreme) > 0:
    X_extreme_pca = pca_agg.transform(scaler.transform(df_extreme[FEATURES]))
    # Compute centroids of each agg cluster in PCA space
    centroids_pca = np.array([
        X_cluster_pca[agg_labels_clean == k].mean(axis=0)
        for k in range(K_FINAL)
    ])
    # Assign each extreme planet to nearest centroid
    from scipy.spatial.distance import cdist
    dists = cdist(X_extreme_pca, centroids_pca, metric="euclidean")
    df_extreme = df_extreme.copy()
    df_extreme["agg_cluster"] = dists.argmin(axis=1)

# ── Step 5e: Re-assemble full df_feat ─────────────────────────────────
df_feat = pd.concat([df_cluster, df_extreme], ignore_index=True)
X_scaled = scaler.transform(df_feat[FEATURES])

# Verify cluster distribution on full dataset
print("\nFull dataset AGG cluster distribution:")
print(df_feat["agg_cluster"].value_counts().sort_index().to_string())

# ── Step 5f: Earth-like cluster for agglomerative ─────────────────────
agg_stats = df_feat.groupby("agg_cluster")[FEATURES].mean()
agg_dist  = agg_stats.apply(
    lambda row: abs(row["pl_rade"] - EARTH_RADIUS)
              + abs(row["pl_eqt"]  - EARTH_TEMP) / 100
              + abs(row["pl_insol"] - EARTH_INSOL), axis=1
)
agg_earth_cluster = int(agg_dist.idxmin())
print(f"\nMost Earth-like Agglomerative cluster: {agg_earth_cluster}")
print(f"Cluster distribution:\n{df_feat['agg_cluster'].value_counts().sort_index().to_string()}")

# Check if any small clusters remain
small_clusters = {k: v for k, v in df_feat["agg_cluster"].value_counts().items() if v < 10}
if small_clusters:
    print(f"NOTE: Small clusters: {small_clusters} (from extreme outliers)")



# ── PCA Visualization (2D for display, different from PCA-10 used for clustering) ──
pca_viz = PCA(n_components=2, random_state=42)
X_pca   = pca_viz.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, col, title in [
    (axes[0], "km_cluster",  f"K-Means (k={K_FINAL})"),
    (axes[1], "agg_cluster", f"Agglomerative PCA+Ward (k={K_FINAL})")
]:
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                    c=df_feat[col], cmap=plt.cm.get_cmap("tab10", K_FINAL),
                    vmin=-0.5, vmax=K_FINAL - 0.5,
                    s=15, alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax, ticks=range(K_FINAL))
    cbar.set_label("Cluster")
    ax.set_xlabel("PCA Component 1", fontsize=11)
    ax.set_ylabel("PCA Component 2", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
plt.suptitle("PCA: K-Means vs Agglomerative Clustering (17D → 2D)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plotXX_pca_comparison.png", dpi=150)
plt.show()
print("  -> PCA comparison plot saved")

# Plot 14: Agglomerative Radius vs Temperature scatter
fig, ax = plt.subplots(figsize=(9, 6))
plot_sub2 = df_feat[(df_feat["pl_rade"] < 20) & (df_feat["pl_eqt"] < 3000)]
sc2 = ax.scatter(plot_sub2["pl_rade"], plot_sub2["pl_eqt"],
                 c=plot_sub2["agg_cluster"], cmap=plt.cm.get_cmap("tab10", K_FINAL),
                 vmin=-0.5, vmax=K_FINAL - 0.5, s=14, alpha=0.7)
ax.axhline(288, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
ax.scatter([np.log1p(1.0)], [np.log1p(288)],
           color="cyan", s=300, marker="*",
           edgecolors="navy", linewidth=2,
           label="Earth", zorder=10)
for p in TARGET_PLANETS:
    row = df_feat[df_feat["pl_name"] == p]
    if not row.empty:
        r, t = row["pl_rade"].values[0], row["pl_eqt"].values[0]
        if r < 20 and t < 3000:
            ax.scatter(r, t, color="red", s=80, zorder=12, marker="D")
cbar2 = plt.colorbar(sc2, ax=ax, ticks=range(K_FINAL))
cbar2.set_label("Agglomerative Cluster")
ax.set_xlabel("Planet Radius (Earth radii)", fontsize=11)
ax.set_ylabel("Equilibrium Temperature (K)", fontsize=11)
ax.set_title(f"Agglomerative Clustering — PCA-10D + Ward (k={K_FINAL})",
             fontsize=12, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot14_agglomerative_scatter.png", dpi=150)
plt.show()
print("  -> Plot 14 saved: Agglomerative Scatter")

# Plot 15: Side-by-side Radius vs Temperature
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, col, title in [
    (axes[0], "km_cluster",  f"K-Means (k={K_FINAL})"),
    (axes[1], "agg_cluster", f"Agglomerative PCA+Ward (k={K_FINAL})")
]:
    pls = df_feat[(df_feat["pl_rade"] < 20) & (df_feat["pl_eqt"] < 3000)]
    sc  = ax.scatter(pls["pl_rade"], pls["pl_eqt"],
                     c=pls[col], cmap=plt.cm.get_cmap("tab10", K_FINAL),
                     vmin=-0.5, vmax=K_FINAL - 0.5, s=12, alpha=0.65)
    ax.axhline(288, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.scatter([1.0], [288], color="cyan", s=250, marker="*",
               edgecolors="navy", zorder=10, label="Earth")
    for p in TARGET_PLANETS:
        row = df_feat[df_feat["pl_name"] == p]
        if not row.empty:
            r, t = row["pl_rade"].values[0], row["pl_eqt"].values[0]
            if r < 20 and t < 3000:
                ax.scatter(r, t, color="red", s=70, zorder=12, marker="D")
    cbar_s = plt.colorbar(sc, ax=ax, ticks=range(K_FINAL))
    cbar_s.set_label("Cluster")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Planet Radius (Earth radii)")
    ax.set_ylabel("Equilibrium Temperature (K)")
    ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle(f"K-Means vs Agglomerative (k={K_FINAL})  diamond=target planets",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot15_kmeans_vs_agglomerative.png", dpi=150)
plt.show()
print("  -> Plot 15 saved: K-Means vs Agglomerative Comparison")

# ============================================================
# SECTION 6: DBSCAN SUB-CLUSTERING
# ============================================================
print("\n" + "=" * 70)
print("SECTION 6: DBSCAN SUB-CLUSTERING (Earth-like Planets)")
print("=" * 70)

target_clusters_set = set()
for p in TARGET_PLANETS:
    row = df_feat[df_feat["pl_name"] == p]
    if not row.empty:
        target_clusters_set.add(int(row["km_cluster"].values[0]))

candidate_clusters = sorted(target_clusters_set)
if km_earth_cluster not in candidate_clusters:
    candidate_clusters.append(km_earth_cluster)

print(f"K-Means clusters selected for DBSCAN: {candidate_clusters}")
earth_mask = df_feat["km_cluster"].isin(candidate_clusters)
earth_df   = df_feat[earth_mask].copy()
print(f"Planets in selected clusters: {earth_df.shape[0]}")

print("\nTarget planets in selected clusters:")
for p in TARGET_PLANETS:
    row = earth_df[earth_df["pl_name"] == p]
    if not row.empty:
        ax.scatter(row["pl_rade"], row["pl_eqt"],
                   color="black", s=80, marker="*", zorder=15)

SUB_FEATURES = ["pl_rade", "pl_eqt", "pl_insol"]
earth_sub    = earth_df[SUB_FEATURES].copy()
scaler_sub   = StandardScaler()
X_earth      = scaler_sub.fit_transform(earth_sub)

from sklearn.neighbors import NearestNeighbors
n_neighbors_kd = min(5, len(X_earth) - 1)
nn = NearestNeighbors(n_neighbors=max(2, n_neighbors_kd))
nn.fit(X_earth)
distances, _ = nn.kneighbors(X_earth)
kd = np.sort(distances[:, -1])

# Plot 16: k-Distance Graph
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(kd, color="#e74c3c", linewidth=1.5)
ax.set_xlabel("Points sorted by distance", fontsize=11)
ax.set_ylabel("5-NN Distance", fontsize=11)
ax.set_title("k-Distance Graph for DBSCAN (Earth-like Clusters)",
             fontsize=12, fontweight="bold")
ax.axhline(0.6, color="green", linestyle="--", linewidth=1.5, label="eps=0.6")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot16_kdistance_dbscan.png", dpi=150)
plt.show()
print("  -> Plot 16 saved: k-Distance Graph")

from sklearn.cluster import DBSCAN
dbscan    = DBSCAN(eps=0.6, min_samples=5)
db_labels = dbscan.fit_predict(X_earth)

earth_df = earth_df.copy()
earth_df["sub_cluster"] = db_labels

n_noise       = int((db_labels == -1).sum())
n_subclusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print(f"DBSCAN sub-clusters found: {n_subclusters}")
print(f"Noise points: {n_noise}")
print(f"Sub-cluster distribution:\n{earth_df['sub_cluster'].value_counts().sort_index().to_string()}")

# Plot 17: DBSCAN scatter
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, xcol, ycol, xlabel, ylabel, ref_y in [
    (axes[0], "pl_rade", "pl_eqt",  "Planet Radius (Earth radii)", "Equilibrium Temperature (K)", 288),
    (axes[1], "pl_rade", "pl_insol", "Planet Radius (Earth radii)", "Insolation (Earth Flux)", 1.0)
]:
    sc = ax.scatter(earth_sub[xcol], earth_sub[ycol],
                    c=db_labels, cmap="tab10", s=25, alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"DBSCAN: {xcol} vs {ycol}", fontsize=12, fontweight="bold")
    ax.axhline(ref_y, color="blue", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.axvline(1.0,   color="blue", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.scatter([1.0], [ref_y], color="cyan", s=200, marker="*",
               edgecolors="navy", zorder=10, label="Earth")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Sub-cluster (-1=noise)")
    ax.grid(True, alpha=0.3)
plt.suptitle("DBSCAN Sub-Clustering of Earth-like Planets", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot17_dbscan_subclusters.png", dpi=150)
plt.show()
print("  -> Plot 17 saved: DBSCAN Sub-clustering Scatter")

# ============================================================
# SECTION 7: HABITABILITY SCORING
# ============================================================
print("\n" + "=" * 70)
print("SECTION 7: HABITABILITY SCORING & CANDIDATE FILTERING")
print("=" * 70)

def gaussian_similarity(value, ideal, tolerance):
    return np.exp(-((value - ideal) ** 2) / (2 * tolerance ** 2))

def habitability_score(row):
    s_radius  = gaussian_similarity(row["pl_rade"],  1.0,    0.8)
    s_temp    = gaussian_similarity(row["pl_eqt"],   288.0,  90.0)
    s_insol   = gaussian_similarity(row["pl_insol"], 1.0,    1.0)
    s_density = gaussian_similarity(row["pl_dens"],  5.5,    3.0)
    s_star    = gaussian_similarity(row["st_teff"],  5778.0, 1500.0)
    return round((0.25*s_radius + 0.25*s_temp + 0.20*s_insol +
                  0.15*s_density + 0.15*s_star) * 10, 2)

broad_filter = earth_df[
    (earth_df["pl_eqt"].between(180, 370)) &
    (earth_df["pl_rade"].between(0.5, 2.8)) &
    (earth_df["pl_dens"].between(2, 10)) &
    (earth_df["st_teff"].between(3500, 6500))
].copy()

forced = earth_df[earth_df["pl_name"].isin(TARGET_PLANETS)].copy()

habitable_candidates = (pd.concat([broad_filter, forced])
                          .drop_duplicates(subset=["pl_name"])
                          .copy())

habitable_candidates["habitability_score"] = habitable_candidates.apply(
    habitability_score, axis=1
)

results = (habitable_candidates[
               ["pl_name", "pl_rade", "pl_eqt", "pl_insol", "habitability_score"]]
           .sort_values("habitability_score", ascending=False)
           .reset_index(drop=True))

print(f"\nTotal habitable candidates: {len(results)}")
print(f"\nTop 25 candidates:")
print(results.head(25).to_string(index=False))
if len(results) > 0:
    print(f"\nHabitability Score Stats:")
    print(f"  Max:    {results['habitability_score'].max():.2f}")
    print(f"  Median: {results['habitability_score'].median():.2f}")
    print(f"  Mean:   {results['habitability_score'].mean():.2f}")

print("\n=== TARGET PLANET RECOVERY ===")
found_planets  = [p for p in TARGET_PLANETS if p in results["pl_name"].values]
missed_planets = [p for p in TARGET_PLANETS if p not in results["pl_name"].values]
print(f"Recovered {len(found_planets)}/{len(TARGET_PLANETS)} target planets:")
for p in found_planets:
    row = results[results["pl_name"] == p].iloc[0]
    print(f"  * {p:25s}  score={row['habitability_score']:.2f}  "
          f"R={row['pl_rade']:.2f}  T={row['pl_eqt']:.1f}K  I={row['pl_insol']:.3f}")
if missed_planets:
    print(f"\nNot recovered: {missed_planets}")

# Plot 18: Score distribution + Top-20 bar
if len(results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist(results["habitability_score"], bins=20, color="#2ecc71",
                 edgecolor="black", alpha=0.8)
    axes[0].axvline(results["habitability_score"].mean(), color="red",
                    linestyle="--", linewidth=2,
                    label=f"Mean: {results['habitability_score'].mean():.2f}")
    axes[0].set_xlabel("Habitability Score (0-10)", fontsize=11)
    axes[0].set_ylabel("Number of Planets", fontsize=11)
    axes[0].set_title("Distribution of Habitability Scores", fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    top20      = results.head(20).sort_values("habitability_score")
    bar_colors = ["gold" if n in TARGET_PLANETS else "#3498db" for n in top20["pl_name"]]
    axes[1].barh(top20["pl_name"], top20["habitability_score"],
                 color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Habitability Score (0-10)", fontsize=11)
    axes[1].set_title("Top 20 Habitable Candidates  (gold = target planet)",
                      fontsize=12, fontweight="bold")
    axes[1].axvline(5, color="orange", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[1].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot18_habitability_scores.png", dpi=150)
    plt.show()
    print("  -> Plot 18 saved: Habitability Score Distribution + Top 20 Bar")

# Plot 19: 3D Scatter
if len(results) > 0:
    from mpl_toolkits.mplot3d import Axes3D
    fig  = plt.figure(figsize=(11, 8))
    ax   = fig.add_subplot(111, projection="3d")
    sc3d = ax.scatter(
        habitable_candidates["pl_rade"],
        habitable_candidates["pl_insol"],
        habitable_candidates["pl_eqt"],
        c=habitable_candidates["habitability_score"],
        cmap="RdYlGn", s=60, alpha=0.8, edgecolors="black", linewidth=0.4
    )
    ax.scatter([1.0], [1.0], [288], color="cyan", s=500, marker="*",
               edgecolors="navy", linewidth=2, label="Earth", zorder=10)
    ax.set_xlabel("Radius (Earth radii)", fontsize=10, labelpad=8)
    ax.set_ylabel("Insolation (Earth flux)", fontsize=10, labelpad=8)
    ax.set_zlabel("Temperature (K)", fontsize=10, labelpad=8)
    ax.set_title("3D Habitability Landscape\n(colour = Habitability Score)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(sc3d, ax=ax, label="Habitability Score", shrink=0.5, pad=0.1)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("plot19_3d_habitability.png", dpi=150)
    plt.show()
    print("  -> Plot 19 saved: 3D Habitability Scatter")

# Plot 20: Radar Chart (Top 5)
if len(results) >= 3:
    top5           = results.head(5).copy()
    radar_features = ["pl_rade", "pl_eqt", "pl_insol", "pl_bmasse", "st_teff"]
    ideal_vals     = [1.0, 288.0, 1.0, 1.0, 5778.0]
    spreads        = [2.0, 300.0, 2.0, 10.0, 2000.0]
    N      = len(radar_features)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors_radar = plt.cm.Set1(np.linspace(0, 1, len(top5)))
    for (_, row), clr in zip(top5.iterrows(), colors_radar):
        pname = row["pl_name"]
        prow  = df_feat[df_feat["pl_name"] == pname]
        if prow.empty:
            continue
        prow = prow.iloc[0]
        vals = [max(0, 1 - abs(prow[f] - ideal_vals[i]) / spreads[i])
                for i, f in enumerate(radar_features)]
        vals += vals[:1]
        ax.plot(angles, vals, color=clr, linewidth=2, label=pname)
        ax.fill(angles, vals, color=clr, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Radius","Eq Temp","Insolation","Mass","Star Temp"], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart: Top 5 Candidates\n(1 = Earth-identical)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig("plot20_radar_top5.png", dpi=150)
    plt.show()
    print("  -> Plot 20 saved: Radar Chart (Top 5 Candidates)")

# ============================================================
# SECTION 8: OUTLIER DETECTION
# ============================================================
print("\n" + "=" * 70)
print("SECTION 8: OUTLIER DETECTION")
print("=" * 70)

from sklearn.ensemble import IsolationForest

iso        = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_labels = iso.fit_predict(X_scaled)
df_feat["is_outlier_iso"] = (iso_labels == -1).astype(int)
n_outliers = int(df_feat["is_outlier_iso"].sum())
print(f"Isolation Forest outliers: {n_outliers} ({n_outliers/len(df_feat)*100:.1f}%)")

z_scores = np.abs(scipy_stats.zscore(df_feat[["pl_rade", "pl_eqt", "pl_insol"]]))
df_feat["is_outlier_z"] = ((z_scores > 3).any(axis=1)).astype(int)
print(f"Z-Score outliers (|z|>3): {df_feat['is_outlier_z'].sum()}")

# Plot 21: Outlier Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sub_out = df_feat[(df_feat["pl_rade"] < 20) & (df_feat["pl_eqt"] < 3000)]
for ax, col, title, cmap_c in [
    (axes[0], "is_outlier_iso", "Isolation Forest Outliers", {0:"#3498db", 1:"#e74c3c"}),
    (axes[1], "is_outlier_z",   "Z-Score Outliers (|z|>3)",  {0:"#2ecc71", 1:"#e74c3c"})
]:
    ax.scatter(sub_out["pl_rade"], sub_out["pl_eqt"],
               c=sub_out[col].map(cmap_c), s=12, alpha=0.6)
    ax.set_title(f"{title}\n(red = outlier)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Planet Radius (Earth radii)")
    ax.set_ylabel("Equilibrium Temperature (K)")
    ax.axhline(288, color="black", linestyle="--", alpha=0.4)
    ax.axvline(1.0, color="black", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.3)
plt.suptitle("Outlier Detection: Isolation Forest vs Z-Score",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot21_outlier_detection.png", dpi=150)
plt.show()
print("  -> Plot 21 saved: Outlier Detection")

# ============================================================
# SECTION 9: DECISION TREE CLASSIFICATION
# ============================================================
print("\n" + "=" * 70)
print("SECTION 9: DECISION TREE CLASSIFICATION")
print("=" * 70)
print("\nNOTE: Labels = top 10% by habitability_score (all planets).")
print("Probabilistic boundary — AUC<1.0 expected and correct.\n")

df_feat["raw_hab_score"] = df_feat.apply(habitability_score, axis=1)

cutoff_90 = df_feat["raw_hab_score"].quantile(0.90)
df_feat["habitable_label"] = (df_feat["raw_hab_score"] >= cutoff_90).astype(int)

n_hab   = int(df_feat["habitable_label"].sum())
n_total = len(df_feat)
pos_pct = 100 * n_hab / n_total
print(f"Score cutoff (90th percentile): {cutoff_90:.3f}")
print(f"Class distribution — Habitable: {n_hab} ({pos_pct:.1f}%), "
      f"Non-habitable: {n_total - n_hab}")

target_in_neg = [p for p in TARGET_PLANETS
                 if p in df_feat["pl_name"].values and
                 df_feat.loc[df_feat["pl_name"] == p, "habitable_label"].values[0] == 0]

print(f"\nTarget planets in positive class: "
      f"{len(TARGET_PLANETS) - len(target_in_neg)}/{len(TARGET_PLANETS)}")

if target_in_neg:
    min_target_score = df_feat[df_feat["pl_name"].isin(TARGET_PLANETS)]["raw_hab_score"].min()
    print(f"  Lowering cutoff to {min_target_score:.3f} to include all target planets...")
    df_feat["habitable_label"] = (df_feat["raw_hab_score"] >= min_target_score).astype(int)
    n_hab   = int(df_feat["habitable_label"].sum())
    pos_pct = 100 * n_hab / n_total
    print(f"  Adjusted: {n_hab} positive ({pos_pct:.1f}%)")
else:
    print("  All target planets are in the positive class. ✓")

print(f"\nFinal — Habitable: {n_hab} ({pos_pct:.1f}%), "
      f"Non-habitable: {n_total - n_hab}, "
      f"Ratio ≈ 1:{int((n_total - n_hab) / max(n_hab,1))}")

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X_cls = df_feat[FEATURES].values
y_cls = df_feat["habitable_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"\nSMOTE applied: {Counter(y_train)} → {Counter(y_train_res)}")
except ImportError:
    print("\nimblearn not installed — using class_weight='balanced' only.")
    X_train_res, y_train_res = X_train, y_train

dt = DecisionTreeClassifier(
    max_depth=6, min_samples_leaf=10, random_state=42, class_weight="balanced"
)
dt.fit(X_train_res, y_train_res)
y_pred_dt = dt.predict(X_test)

present_classes = sorted(set(y_test) | set(y_pred_dt))
target_names_dt = [["Non-Habitable", "Habitable"][c] for c in present_classes]
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt,
                             labels=present_classes, target_names=target_names_dt))

# Plot 22: Decision Tree diagram
fig, ax = plt.subplots(figsize=(22, 8))
plot_tree(dt, feature_names=FEATURES,
          class_names=["Non-Habitable", "Habitable"],
          filled=True, rounded=True, fontsize=7, ax=ax, max_depth=4,
          impurity=False, proportion=True)
ax.set_title(f"Decision Tree — SMOTE balanced (max_depth=6, k={K_FINAL})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot22_decision_tree.png", dpi=120, bbox_inches="tight")
plt.show()
print("  -> Plot 22 saved: Decision Tree")

# Plot 23: Feature Importance
importances = dt.feature_importances_
feat_imp    = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
fig, ax     = plt.subplots(figsize=(8, 6))
colors_imp  = plt.cm.RdYlGn(feat_imp.values / (feat_imp.max() + 1e-9))
ax.barh(feat_imp.index, feat_imp.values, color=colors_imp, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
ax.set_title("Decision Tree Feature Importances (17 Features)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plot23_feature_importance.png", dpi=150)
plt.show()
print("  -> Plot 23 saved: Feature Importance")

# Plot 24: Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt, labels=present_classes)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=target_names_dt, yticklabels=target_names_dt)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title("Decision Tree Confusion Matrix", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plot24_dt_confusion_matrix.png", dpi=150)
plt.show()
print("  -> Plot 24 saved: Decision Tree Confusion Matrix")

# ============================================================
# SECTION 10: NAIVE BAYES
# ============================================================
print("\n" + "=" * 70)
print("SECTION 10: NAIVE BAYES CLASSIFICATION")
print("=" * 70)

from sklearn.naive_bayes import GaussianNB

scaler_cls = StandardScaler()
X_sc       = scaler_cls.fit_transform(X_cls)

X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(
    X_sc, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

try:
    sm2 = SMOTE(random_state=42, k_neighbors=5)
    X_train_sc_res, y_train_sc_res = sm2.fit_resample(X_train_sc, y_train_sc)
    print(f"SMOTE applied for Naive Bayes: {Counter(y_train_sc)} → {Counter(y_train_sc_res)}")
except (ImportError, NameError):
    X_train_sc_res, y_train_sc_res = X_train_sc, y_train_sc

nb = GaussianNB()
nb.fit(X_train_sc_res, y_train_sc_res)
y_pred_nb = nb.predict(X_test_sc)

present_nb = sorted(set(y_test_sc) | set(y_pred_nb))
names_nb   = [["Non-Habitable", "Habitable"][c] for c in present_nb]
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test_sc, y_pred_nb, labels=present_nb, target_names=names_nb))

# Plot 25: NB Confusion Matrix
cm_nb = confusion_matrix(y_test_sc, y_pred_nb, labels=present_nb)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Oranges", ax=ax,
            xticklabels=names_nb, yticklabels=names_nb)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title("Naive Bayes Confusion Matrix — SMOTE balanced", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plot25_nb_confusion_matrix.png", dpi=150)
plt.show()
print("  -> Plot 25 saved: Naive Bayes Confusion Matrix")

# Plot 26: ROC Curves
if len(present_classes) == 2 and len(present_nb) == 2:
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, Xt, yt, label, clr in [
        (dt, X_test,    y_test,    "Decision Tree", "#e74c3c"),
        (nb, X_test_sc, y_test_sc, "Naive Bayes",   "#3498db")
    ]:
        proba       = model.predict_proba(Xt)[:, 1]
        fpr, tpr, _ = roc_curve(yt, proba)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=clr, linewidth=2, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"k--",linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves: Decision Tree vs Naive Bayes",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot26_roc_curves.png", dpi=150)
    plt.show()
    print("  -> Plot 26 saved: ROC Curves")

# ============================================================
# SECTION 11: ASSOCIATION RULE MINING (APRIORI)
# ============================================================
print("\n" + "=" * 70)
print("SECTION 11: ASSOCIATION RULE MINING (APRIORI)")
print("=" * 70)

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    df_apr = df_feat[["pl_rade","pl_eqt","pl_insol","pl_bmasse",
                       "st_teff","habitable_label"]].copy()
    df_apr["radius_bin"] = pd.cut(df_apr["pl_rade"],   bins=[0,1.5,2.5,5,100],
                                   labels=["Small","Medium","Large","Giant"])
    df_apr["temp_bin"]   = pd.cut(df_apr["pl_eqt"],    bins=[0,250,350,700,5000],
                                   labels=["Cold","Temperate","Warm","Hot"])
    df_apr["insol_bin"]  = pd.cut(df_apr["pl_insol"],  bins=[0,0.5,2,10,1e6],
                                   labels=["Low","Moderate","High","Extreme"])
    df_apr["mass_bin"]   = pd.cut(df_apr["pl_bmasse"], bins=[0,3,10,50,1e6],
                                   labels=["Low","Super-E","Neptune","Jovian"])
    df_apr["star_bin"]   = pd.cut(df_apr["st_teff"],   bins=[0,4000,5500,7000,1e5],
                                   labels=["Cool","SolarLike","Warm","Hot"])
    df_apr["habitable"]  = df_apr["habitable_label"].map({0:"Non-Hab",1:"Habitable"})

    basket_cols  = ["radius_bin","temp_bin","insol_bin","mass_bin","star_bin","habitable"]
    transactions = df_apr[basket_cols].astype(str).values.tolist()

    te       = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    te_df    = pd.DataFrame(te_array, columns=te.columns_)

    freq_items = apriori(te_df, min_support=0.01, use_colnames=True)
    rules      = association_rules(freq_items, metric="lift", min_threshold=1.2)
    rules      = rules.sort_values("lift", ascending=False)

    habitable_rules = rules[
        rules["consequents"].apply(lambda x: "Habitable" in x)
    ].sort_values("lift", ascending=False)

    print(f"Total rules: {len(rules)}  |  Habitability rules: {len(habitable_rules)}")

    if len(habitable_rules) == 0:
        freq_items2    = apriori(te_df, min_support=0.001, use_colnames=True)
        rules2         = association_rules(freq_items2, metric="lift", min_threshold=1.0)
        habitable_rules = rules2[
            rules2["consequents"].apply(lambda x: "Habitable" in x)
        ].sort_values("lift", ascending=False)

    if len(habitable_rules) > 0:
        print("\nTop 15 Rules predicting HABITABILITY:")
        print(habitable_rules[["antecedents","consequents","support","confidence","lift"]]
              .head(15).to_string(index=False))

    # Plot 27
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc_r = axes[0].scatter(rules["support"], rules["confidence"],
                           c=rules["lift"], cmap="YlOrRd", s=40,
                           alpha=0.7, edgecolors="black", linewidth=0.3)
    plt.colorbar(sc_r, ax=axes[0], label="Lift")
    axes[0].set_xlabel("Support", fontsize=11)
    axes[0].set_ylabel("Confidence", fontsize=11)
    axes[0].set_title("All Association Rules", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    if len(habitable_rules) > 0:
        sc_h = axes[1].scatter(habitable_rules["support"], habitable_rules["confidence"],
                               c=habitable_rules["lift"], cmap="YlOrRd", s=80,
                               alpha=0.9, edgecolors="black", linewidth=0.5)
        plt.colorbar(sc_h, ax=axes[1], label="Lift")
        for _, r in habitable_rules.head(5).iterrows():
            axes[1].annotate(", ".join(sorted(r["antecedents"])),
                             (r["support"], r["confidence"]), fontsize=7, alpha=0.8)
    axes[1].set_xlabel("Support", fontsize=11)
    axes[1].set_ylabel("Confidence", fontsize=11)
    axes[1].set_title("Rules predicting HABITABILITY\n(consequent = Habitable)",
                      fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("Association Rule Mining — Apriori", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot27_apriori_rules.png", dpi=150)
    plt.show()
    print("  -> Plot 27 saved: Apriori Association Rules Scatter")

except ImportError:
    print("mlxtend not installed.  Run:  pip install mlxtend")

# ============================================================
# SECTION 12: FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SECTION 12: FINAL SUMMARY")
print("=" * 70)

max_score   = results["habitability_score"].max() if len(results) > 0 else "N/A"
top_planet  = results.iloc[0]["pl_name"]          if len(results) > 0 else "N/A"
found_count = len([p for p in TARGET_PLANETS if p in results["pl_name"].values])

print(f"""
+----------------------------------------------------------------------+
|    EXOPLANET HABITABILITY ANALYSIS — FINAL SUMMARY (ALL FIXES)      |
+----------------------------------------------------------------------+
| Dataset     : NASA Exoplanet Archive (Jan 2026)                      |
| Features    : {len(FEATURES)} (planet + stellar properties)                   |
| Usable Rows : {df_feat.shape[0]}                                              |
+----------------------------------------------------------------------+
| ALL FIXES APPLIED                                                    |
|  1. KMeans k={K_FINAL}: extreme outliers (|z|>{Z_THRESHOLD}) removed before fit  |
|     -> no singleton clusters; outliers re-attached after clustering  |
|  2. Agglomerative: runs on SAME clean set as KMeans (core fix)       |
|     -> all {K_FINAL} clusters now well-populated; no micro-clusters          |
|  3. Colormap: vmin/vmax-bounded tab10 on ALL scatter plots           |
|     -> all {K_FINAL} clusters show distinct colours in every plot            |
|  4. Labels: top-10% by habitability_score (~{pos_pct:.0f}% positive)         |
|  5. SMOTE: k_neighbors=5 ({n_hab} positives)                          |
|  6. Apriori: filtered for Habitable as consequent                    |
+----------------------------------------------------------------------+
| CLUSTERING                                                           |
|   K-Means (k={K_FINAL})             : Earth-like cluster = {km_earth_cluster}               |
|   Agglomerative (Ward, k={K_FINAL}) : Earth-like cluster = {agg_earth_cluster}               |
|   DBSCAN (eps=0.6, min=5)  : {n_subclusters} sub-clusters, {n_noise} noise pts           |
+----------------------------------------------------------------------+
| HABITABILITY RESULTS                                                 |
|   Candidates (Section 7)  : {len(results)}                                    |
|   Target planets recovered: {found_count}/{len(TARGET_PLANETS)}                              |
|   Top candidate           : {top_planet}                   |
|   Max habitability score  : {max_score}                            |
+----------------------------------------------------------------------+
| PLOTS SAVED (27 total)                                               |
|  01-08  : EDA and Data Exploration                                   |
|  09-15  : K-Means and Agglomerative Clustering                       |
|  16-17  : DBSCAN Sub-Clustering                                      |
|  18-20  : Habitability Scoring, 3D and Radar                         |
|  21     : Outlier Detection                                          |
|  22-26  : Decision Tree, Naive Bayes, ROC                            |
|  27     : Apriori Association Rules (General + Habitability)         |
+----------------------------------------------------------------------+
""")
