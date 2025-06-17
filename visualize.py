import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and parse the Apple Health export XML
tree = ET.parse('/Users/sujiththota/Downloads/apple_health_export/export.xml')  # Replace with full path if needed
root = tree.getroot()

# List to hold structured records
records = []

# Loop through each <Record> in the XML
for record in root.findall('Record'):
    records.append({
        'type': record.attrib.get('type'),
        'sourceName': record.attrib.get('sourceName'),
        'sourceVersion': record.attrib.get('sourceVersion'),
        'unit': record.attrib.get('unit'),
        'creationDate': record.attrib.get('creationDate'),
        'startDate': record.attrib.get('startDate'),
        'endDate': record.attrib.get('endDate'),
        'value': record.attrib.get('value')
    })

# Convert to DataFrame
df = pd.DataFrame(records)

# Preview the first few rows
print(df.head())

# Optional: Save to CSV
df.to_csv("apple_health_data.csv", index=False)



# Convert dates
df['startDate'] = pd.to_datetime(df['startDate'])
df['endDate'] = pd.to_datetime(df['endDate'])
df['value'] = pd.to_numeric(df['value'], errors='coerce')


# Filter step count
steps_df = df[df['type'] == 'HKQuantityTypeIdentifierStepCount']

# Group by day
steps_daily = steps_df.groupby(steps_df['startDate'].dt.date)['value'].sum().reset_index()

# Plot
plt.figure(figsize=(12, 4))
sns.lineplot(data=steps_daily, x='startDate', y='value')
plt.title('📊 Daily Step Count')
plt.xlabel('Date')
plt.ylabel('Steps')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Filter heart rate
hr_df = df[df['type'] == 'HKQuantityTypeIdentifierHeartRate']

# Group by day
hr_daily = hr_df.groupby(hr_df['startDate'].dt.date)['value'].mean().reset_index()

# Plot
plt.figure(figsize=(12, 4))
sns.lineplot(data=hr_daily, x='startDate', y='value', color='red')
plt.title('❤️ Average Daily Heart Rate')
plt.xlabel('Date')
plt.ylabel('BPM')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sleep_df = df[df['type'] == 'HKCategoryTypeIdentifierSleepAnalysis'].copy()
sleep_df['duration_hours'] = (sleep_df['endDate'] - sleep_df['startDate']).dt.total_seconds() / 3600

# Group by date
sleep_daily = sleep_df.groupby(sleep_df['startDate'].dt.date)['duration_hours'].sum().reset_index()

# Plot
plt.figure(figsize=(12, 4))
sns.barplot(data=sleep_daily, x='startDate', y='duration_hours', color='purple')
plt.title('🛌 Sleep Duration Per Day')
plt.xlabel('Date')
plt.ylabel('Hours')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
