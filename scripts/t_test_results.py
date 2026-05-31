from scipy import stats

t_statistic = 4.035
df = 4

# Two-tailed p-value
# sf = survival function = 1 - CDF, gives the one-tailed p-value
# multiply by 2 for two-tailed
p_value = stats.t.sf(abs(t_statistic), df) * 2

print(f"t-statistic: {t_statistic}")
print(f"degrees of freedom: {df}")
print(f"p-value (two-tailed): {p_value:.4f}")
