import pandas as pd

import pandas as pd

# Path to your Excel file
excel_file = r"D:\Hackathon\DeepReef-Ai\global_bleaching_environmental(2).xlsx"

# Read Excel
df = pd.read_excel(excel_file)

# Save as CSV
csv_file = r"D:\Hackathon\DeepReef-Ai\global_bleaching_environmental(2).csv"
df.to_csv(csv_file, index=False)

print("✅ Excel file converted to CSV successfully!")

