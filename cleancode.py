import pandas as pd

# Full file path
file_path = r"D:\Hackathon\DeepReef-Ai\cleaned_data.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# 🔹 Delete rows where a specific column has "nd"
# Replace 'ColumnName' with the actual column name you want to check
df = df[df['Bleaching_Comments'] != "nd"]

# Save the cleaned dataset (you can choose .csv or .xlsx)
df.to_csv(r"D:\Hackathon\DeepReef-Ai\cleaned_data2.csv", index=False)
# OR
# df.to_excel(r"D:\Hackathon\DeepReef-Ai\cleaned_data.xlsx", index=False)

print("✅ Cleaning complete. File saved.")

