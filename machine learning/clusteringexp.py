import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
data = {
    'Bread':   [1, 1, 0, 1, 0],
    'Butter':  [1, 1, 0, 1, 0],
    'Milk':    [1, 0, 0, 1, 0],
    'Diapers': [0, 0, 1, 1, 1],
    'Beer':    [0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
#print(df)
# min_support = 2 transactions out of 5 = 2/5 = 0.4
frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.7
)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
