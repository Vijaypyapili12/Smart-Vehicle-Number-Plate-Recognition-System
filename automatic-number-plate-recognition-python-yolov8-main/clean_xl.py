import pandas as pd

# Path to the Excel file
file_path = "vehicle_entries.xlsx"

# Create an empty DataFrame with the same columns
df = pd.DataFrame(columns=["Number Plate", "Timestamp", "Slot Number"])

# Overwrite the file
df.to_excel(file_path, index=False)

print("All data cleared from vehicle_entries.xlsx")
