import requests
import os

def download_csv(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'w') as f:
            f.write(response.text)
        print(f"CSV file '{file_name}' downloaded successfully")
    else:
        print(f"Failed to download CSV file from '{url}'")

# URLs and file names
urls_and_filenames = [
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv', 'FuelConsumptionCo2.csv'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/china_gdp.csv', 'china_gdp.csv')
]

# Download CSV files
for url, filename in urls_and_filenames:
    download_csv(url, filename)

