import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# ---------------------------------------------------------
# 1. Load Trained Pipeline Models
# ---------------------------------------------------------
model_fracture = load("best_model1_VotingRegressor.joblib")        # Fracture Toughness Model
model_youngs = load("best_model_KNeighborsRegressor.joblib")     # Young's Modulus Model

# ---------------------------------------------------------
# 2. Generate a Large Synthetic Parameter Space (Grid)
# ---------------------------------------------------------
graphene_range = np.linspace(0, 100, 101)          # From 0% to 100% with step size 1
defect_density_range = np.linspace(0, 5, 51)       # From 0 to 5 with step size 0.1
defect_types = ['SV', 'DV', 'SW', 'SH', 'CH']      # Unique defect types

grid_list = []
for g in graphene_range:
    for d in defect_density_range:
        for dt in defect_types:
            grid_list.append({
                'Graphene': g,
                'Defect Density': d,
                'Defect Type': dt
            })

X_grid = pd.DataFrame(grid_list)
print(f"Total synthetic configurations generated for evaluation: {len(X_grid):,}")

# ---------------------------------------------------------
# 3. High-Throughput Predictions via ML Models
# ---------------------------------------------------------
X_grid['Pred_Fracture_Toughness'] = model_fracture.predict(X_grid)
X_grid['Pred_Youngs_Modulus'] = model_youngs.predict(X_grid)

# ---------------------------------------------------------
# 4. Pareto Front Identification Algorithm
# ---------------------------------------------------------
def identify_pareto(df, col1, col2):
    """
    Identifies Pareto-optimal points maximizing both target columns (col1 and col2).
    """
    scores = df[[col1, col2]].values
    pareto_mask = np.ones(scores.shape[0], dtype=bool)
    
    for i, score in enumerate(scores):
        if pareto_mask[i]:
            pareto_mask[pareto_mask] = np.any(scores[pareto_mask] > score, axis=1)
            pareto_mask[i] = True
            
    return pareto_mask

pareto_mask = identify_pareto(X_grid, 'Pred_Youngs_Modulus', 'Pred_Fracture_Toughness')
pareto_df = X_grid[pareto_mask].sort_values(by='Pred_Youngs_Modulus')

print(f"\nNumber of Pareto-optimal configurations found: {len(pareto_df)}")
print("\nSample Pareto Optimal Structures:")
print(pareto_df[['Graphene', 'Defect Density', 'Defect Type', 'Pred_Youngs_Modulus', 'Pred_Fracture_Toughness']].head(10))

# Export optimal designs to CSV
pareto_df.to_csv("optimal_pareto_structures.csv", index=False)

# ---------------------------------------------------------
# 5. Compact, Publication-Ready Pareto Plot Generation
# ---------------------------------------------------------
# Reduced figure size for a more compact presentation (6x4.5 inches)
plt.figure(figsize=(4, 2), dpi=300)

# 1. Scatter plot of all evaluated ML points
plt.scatter(X_grid['Pred_Youngs_Modulus'], X_grid['Pred_Fracture_Toughness'], 
            c='gainsboro', alpha=0.1, s=2, label='ML Parameter Space')

# 2. Highlight Pareto Front Optimal Points
plt.scatter(pareto_df['Pred_Youngs_Modulus'], pareto_df['Pred_Fracture_Toughness'], 
            c='crimson', s=5, zorder=3, label='Pareto Optimal Front')

plt.plot(pareto_df['Pred_Youngs_Modulus'], pareto_df['Pred_Fracture_Toughness'], 
         c='crimson', linestyle='--', linewidth=1.2, zorder=2)

# Compact labels & layout styling
plt.xlabel("Young's Modulus (GPa)", fontsize=4, fontweight='bold')
plt.ylabel("Fracture Toughness (MJ/m³)", fontsize=4, fontweight='bold')
plt.title("Multi-Objective Pareto Optimization", fontsize=4, fontweight='bold')
plt.legend(loc='lower left', frameon=True, fontsize=4)
plt.grid(True, linestyle=':', alpha=0.1)

plt.tight_layout()
plt.savefig("Pareto_Front_Plot.png", dpi=300)
plt.show()