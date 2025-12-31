## @file farmers_market_apriori.py
## @brief Load farmers market dataset, clean it, run Apriori algorithm, generate rules, and plot top associations.
## @author Shubhankar Dumka
## @date 2025-11-06

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)  # no truncation for cell content

## @brief Load the raw farmers market dataset.
## @return DataFrame containing the raw dataset
df = pd.read_csv("../datasets/farmers_market.csv")

## @brief Columns to keep for analysis
cols_to_keep = ['Market_ID', 'Market_Name', 'Is_Organic', 'Is_Baked_Goods', 'Is_Cheese', 
                'Is_Crafts', 'Is_Flowers', 'Is_Eggs', 'Is_Seafood', 'Is_Herbs', 'Is_Vegetables',
                'Is_Honey', 'Is_Jams', 'Is_Maple', 'Is_Meat', 'Is_Nursery', 'Is_Nuts',
                'Is_Plants', 'Is_Poultry', 'Is_Prepared', 'Is_Soap', 'Is_Trees', 'Is_Wine',
                'Is_Coffee', 'Is_Beans', 'Is_Fruits', 'Is_Grains', 'Is_Juices', 'Is_Mushrooms',
                'Is_Pet_Food', 'Is_Tofu', 'Is_Wild_Harvested']
df = df[cols_to_keep]

## @brief Columns representing products
product_columns = cols_to_keep[2:]  # Skip Market_ID and Market_Name

## @brief Clean dataset: fill missing values and convert product columns to boolean
df[product_columns] = df[product_columns].fillna(False).astype(bool)

## @brief Drop rows where no products are sold
df = df[df[product_columns].any(axis=1)].reset_index(drop=True)

## @brief Save cleaned dataset
df.to_csv('cleaned_farmers_market.csv', index=False)

## @brief Prepare product-only DataFrame for Apriori
dfProducts = df[product_columns]

## @brief Run Apriori algorithm
## @param dfProducts DataFrame with boolean product columns
## @param min_support Minimum support threshold
## @return DataFrame of frequent itemsets
frequentItemsets = apriori(dfProducts, min_support=0.5, use_colnames=True)

## @brief Save frequent itemsets to a file
with open('FrequentItemsets.obj', 'wb') as fh:
    pickle.dump(frequentItemsets, fh)

## @brief Load frequent itemsets from file and inspect
with open('FrequentItemsets.obj', 'rb') as fh:
    frequentItemsets_loaded = pickle.load(fh)
print(frequentItemsets_loaded)

## @brief Ensure itemsets are frozensets for rule generation
frequentItemsets['itemsets'] = frequentItemsets['itemsets'].apply(lambda x: frozenset(x))

## @brief Generate association rules
## @param frequentItemsets DataFrame with frequent itemsets
## @param metric Metric to use (confidence)
## @param min_threshold Minimum confidence
## @param num_itemsets Number of itemsets (optional)
## @return DataFrame containing association rules
rules = association_rules(frequentItemsets, metric='confidence', min_threshold=0.5, num_itemsets=dfProducts.shape[0])

## @brief Save rules to a file
with open('./Rules.obj', 'wb') as fh:
    pickle.dump(rules, fh)

## @brief Sort rules by confidence in descending order and print
sorted_rules = rules.sort_values(by='confidence', ascending=False)
print(sorted_rules)

## @brief Product to analyze for top co-occurring items
pProductList = ['Is_Baked_Goods']
inputProduct = 'Is_Baked_Goods'

## @brief Filter rules where the product is in the antecedent
filtered_rules = rules[rules["antecedents"].apply(lambda x: inputProduct in x)]

if filtered_rules.empty:
    print(f"No rules found for product: {inputProduct}")
else:
    ## @brief Group by antecedents and consequents, take max confidence
    grouped = filtered_rules.groupby(['antecedents', 'consequents'])[['confidence']].max()

    ## @brief Sort by confidence descending and take top 10
    top_rules = grouped.sort_values(by='confidence', ascending=False).head(10).copy()

    ## @brief Convert frozensets to readable strings for x-axis labels
    top_rules.index = top_rules.index.map(
        lambda x: f"{', '.join(list(x[0]))} -> {', '.join(list(x[1]))}"
    )

    ## @brief Plot top association rules
    ax = top_rules.plot(kind='bar', legend=False, figsize=(10,6))
    ax.invert_xaxis()  # invert x-axis
    plt.title('Top Products likely sold with ' + inputProduct)
    plt.ylabel('Confidence')
    plt.xlabel('Antecedent -> Consequent')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()
