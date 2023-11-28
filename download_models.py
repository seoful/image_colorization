import requests
from urllib.parse import urlencode
import os

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

def download_file(url, directory, name):
        
    final_url = base_url + urlencode(dict(public_key=url))
    response = requests.get(final_url)
    download_url = response.json()['href']
    print("Download from " + url)

    download_response = requests.get(download_url)
    print("Saving")


    abs_lib = os.path.abspath('')
    path = abs_lib + directory

    try:
        os.mkdir(abs_lib + '/models')
    except OSError as error:  
        pass

    try:  
        os.mkdir(path)  
    except OSError as error:  
        pass

    with open(name, 'wb') as f:
        print(f)
        f.write(download_response.content)

def download():
    download_file('https://disk.yandex.ru/d/ucq08radoZH2pw','/models/lab','/models/lab/gan_model.pth')
    download_file('https://disk.yandex.ru/d/7PyG8akR2nEheQ','/models/lab','models/lab/res18-unet.pt')
    download_file('https://disk.yandex.ru/d/WTkZeLIJTd-z2Q','/models/lab','/models/lab/res_net_unet_gan.pt')
    download_file('https://disk.yandex.ru/d/BA73KtYX6ILyYQ','/models/oklab','/models/oklab/gan_model.pth')
    download_file('https://disk.yandex.ru/d/mgcwrJephSZ-wg','/models/oklab','/models/oklab/res18-unet.pt')
    download_file('https://disk.yandex.ru/d/896aEGzTnfOfMQ','/models/oklab','/models/oklab/res_net_unet_gan.pt')

if __name__ == '__main__':
    download()
