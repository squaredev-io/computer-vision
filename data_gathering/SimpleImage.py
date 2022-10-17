# SimpleImage is a library used for downloading images based on keywords mostly from Google. 

# pip install --upgrade simple_image_download

from simple_image_download import simple_image_download as sp
response = sp.simple_image_download()

#Set the directory in which the images will be downloaded.
#After the completion you will find there a folder with the name "simple_images"
response.directory="set_your_directory/"

#Set the keyword(s) and the number of images. The machine will download the limit number of images per keyword.
response.download(keywords="keyword1,keyword2,....",limit=1000)
