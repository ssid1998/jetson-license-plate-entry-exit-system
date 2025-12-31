from itertools import combinations
def generate_candidates(L, k):
    return set([frozenset(sorted(set(i) | set(j))) for i in L for j in L if len(set(i) |
set(j)) == k])
def apriori(transactions, min_support):
    items = set(item for transaction in transactions for item in transaction)
    itemset = set(frozenset([item]) for item in items)
    L = [set()]
    k = 1
    while itemset:
        counts = {i: sum([i.issubset(transaction) for transaction in transactions]) for i in itemset}
        L_k = set(i for i in counts if counts[i] >= min_support)
        L.append(L_k)
        k += 1
        itemset = generate_candidates(L_k, k)
        return L[1:]
# Dataset from Table 1
transactions = [
{'Bread', 'Butter', 'Milk'},
{'Bread', 'Milk'},
{'Butter', 'Milk'},
{'Bread', 'Butter'},
{'Bread', 'Cheese', 'Butter'},
{'Milk', 'Cheese'},
{'Cheese'},
{'Bread', 'Milk', 'Butter'}
]
min_support = 4 # 50% of 8 transactions
frequent_itemsets = apriori(transactions, min_support)
print("Frequent Itemsets:")
for k, itemsets in enumerate(frequent_itemsets, 1):
    print(f"{k}-itemsets:")
for itemset in itemsets:
    print(f" {set(itemset)}")