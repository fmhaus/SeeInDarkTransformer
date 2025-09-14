import requests
import os
from tqdm import tqdm

DATASET_PATH = './../dataset/'

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), 'Downloading...'):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_sony():
    os.makedirs(DATASET_PATH)
    zip_file = os.path.join(DATASET_PATH, 'Sony.zip')
    download_file_from_google_drive('10kpAcvldtcb9G2ze5hTcF1odzu4V_Zvh', zip_file)
    print('Unzipping...')
    os.system(f'unzip {zip_file} -d {DATASET_PATH}')

def download_fuji():
    os.makedirs(DATASET_PATH)
    zip_file = os.path.join(DATASET_PATH, 'Fuji.zip')
    download_file_from_google_drive('12hvKCjwuilKTZPe9EZ7ZTb-azOmUA3HT', zip_file)
    print('Unzipping...')
    os.system(f'unzip {zip_file} -d {DATASET_PATH}')
    
download_sony()
#download_fuji()
