import requests

def download_csv(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'w') as f:
            f.write(response.text)
        print(f"CSV file '{file_name}' downloaded successfully")
    else:
        print(f"Failed to download CSV file from '{url}'")

urls_and_filenames = [
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv','Weather_Data.csv')
]

for url, filename in urls_and_filenames:
    download_csv(url, filename)