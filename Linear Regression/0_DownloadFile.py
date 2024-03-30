import requests
import os

# URL of the CSV file
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'

# Get the current working directory
current_directory = os.getcwd()

# Specify the path where you want to save the file
file_path = os.path.join(current_directory, 'FuelConsumptionCo2.csv')

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the content of the response (the CSV data)
    csv_data = response.text

    # Save the CSV data to a file
    with open('FuelConsumptionCo2.csv', 'w') as f:
        f.write(csv_data)

    print("CSV file downloaded successfully")

else:
    print("Failed to download CSV file")
