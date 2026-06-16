from scipy import stats
import numpy as np

# Plug in the exact 5-fold Macro F1 scores (at the 0.5 threshold)
baseline_scores = np.array([66.05, 66.14, 65.38, 66.87, 66.98])
zero_shot_scores = np.array([65.88,	68.88, 69.68,	66.90, 70.41])
few_shot_v3_scores = np.array([65.91,	69.91, 69.89, 67.25, 69.95])

print("=== PAIRED T-TEST RESULTS ===")

# 1. Few-Shot V3 vs. Baseline
t_stat_v3, p_val_v3 = stats.ttest_rel(few_shot_v3_scores, baseline_scores)
mean_diff_v3 = np.mean(few_shot_v3_scores - baseline_scores)

print(f"\nFew-Shot vs. Baseline:")
print(f"Mean Difference: {mean_diff_v3:+.2f}")
print(f"t-statistic: {t_stat_v3:.4f}")
print(f"p-value (two-tailed): {p_val_v3:.4f}")

# 2. Zero-Shot vs. Baseline
t_stat_zs, p_val_zs = stats.ttest_rel(zero_shot_scores, baseline_scores)
mean_diff_zs = np.mean(zero_shot_scores - baseline_scores)

print(f"\nZero-Shot vs. Baseline:")
print(f"Mean Difference: {mean_diff_zs:+.2f}")
print(f"t-statistic: {t_stat_zs:.4f}")
print(f"p-value (two-tailed): {p_val_zs:.4f}")

# 3. Few-Shot V3 vs. Zero-Shot
t_stat_comp, p_val_comp = stats.ttest_rel(few_shot_v3_scores, zero_shot_scores)
mean_diff_comp = np.mean(few_shot_v3_scores - zero_shot_scores)

print(f"\nFew-Shot vs. Zero-Shot:")
print(f"Mean Difference: {mean_diff_comp:+.2f}")
print(f"t-statistic: {t_stat_comp:.4f}")
print(f"p-value (two-tailed): {p_val_comp:.4f}")
