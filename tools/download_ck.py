import gdown
import os

def download_file_from_google_drive(file_url, destination):
 
    file_id = file_url.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?id={file_id}'

    gdown.download(download_url, destination, quiet=False)


pretrained_folder = "pretrained"
if not os.path.exists(pretrained_folder):
    os.makedirs(pretrained_folder)

file_url_CoDA = 'https://drive.google.com/file/d/1lF_tDzJEg9ruVqKMwPZSxnnRa_Q6p_sy/view?usp=sharing'
file_url_Pre = 'https://drive.google.com/file/d/1BqHFLhtFTewsM4xLixnahCjkpReevm6k/view?usp=sharing'


destination_CoDA = os.path.join(pretrained_folder, "CoDA.pth") 
destination_Pre = os.path.join(pretrained_folder, "Pre.pth")

# 下载文件
download_file_from_google_drive(file_url_CoDA, destination_CoDA)
download_file_from_google_drive(file_url_Pre, destination_Pre)
