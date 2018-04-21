import os
import urllib.request
import zipfile

DATASET_URL = 'http://www.selameab.com/downloads/bikes_ds.zip'
TEMP_ZIP = 'bikes_ds.zip'

if __name__ == '__main__':
    # Download zip if it doesn't exist
    print('Downloading', DATASET_URL, ' - ', end='')
    if not os.path.isfile(TEMP_ZIP):
        urllib.request.urlretrieve(DATASET_URL, TEMP_ZIP)
        print('Done')
    else:
        print('File Exists')

    # Extract zip
    print('Extracting zip - ', end='')
    with zipfile.ZipFile(TEMP_ZIP, 'r') as zip_ref:
        zip_ref.extractall('')
        print('Done')
