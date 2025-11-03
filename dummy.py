import pandas as pd

# ====== Step 1: Load your dataset ======
file_path = r"C:\Users\shobika\New folder\billing data.csv"

if file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
elif file_path.endswith(".xlsx"):
    df = pd.read_excel(file_path)
else:
    raise ValueError("Unsupported file type. Use .csv or .xlsx")

print(f"Original dataset rows: {len(df)}\n")

# ====== Step 2: Identify categorical columns ======
# You can specify columns manually or detect object-type columns automatically
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns detected: {categorical_cols}\n")

# ====== Step 3: Get distinct values for each categorical column ======
distinct_data = {}
for col in categorical_cols:
    distinct_values = df[col].dropna().unique().tolist()
    distinct_data[col] = distinct_values
    print(f"Distinct values for column '{col}':")
    print(distinct_values)
    print("------\n")

# ====== Step 4: Optional - Save distinct values to a new CSV ======
# Flatten into a DataFrame with column name and distinct value
rows = []
for col, values in distinct_data.items():
    for val in values:
        rows.append({"column": col, "distinct_value": val})

df_distinct_values = pd.DataFrame(rows)
df_distinct_values.to_csv("categorical_distinct_values.csv", index=False)
print("Distinct values saved to 'categorical_distinct_values.csv'")
