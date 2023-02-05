import cv2
import pytesseract
import pandas as pd
import os
from preprocessing import preprocess_image, preprocess_txt
import pickle

df = pd.read_csv("meme_fb_memotion.csv")

try:
    from PIL import Image
except ImportError:
    import Image

data = {"image": [], "filepath": [], "text": [], "label": []}

# Iterate over the rows in the DataFrame
for index, row in df.iterrows():
    # Get the label and image values for the current row
    label = row['label']
    image = row['image']
    # Check the label value and append the image to the corresponding list
    try:
        if label == 'offensive':
            data['label'].append(int(1))
        if label == 'not_offensive':
            data['label'].append(int(0))

        data['image'].append(
            preprocess_image(
                os.path.join('./images/', image)))

        data['filepath'].append(
            os.path.join('./images/', image))

        data['text'].append(
            preprocess_txt(
                pytesseract.image_to_string(
                    Image.open(
                        os.path.join('./images/', image)))))

    except Exception as e:
        print(e)
        continue

    print('\r images: {} texts: {} labels : {}'.format(len(data['image']), len(data['text']), len(data['label'])), end='')

with open("data.pkl", "wb") as pickle_out:
    pickle.dump(data, pickle_out)
pickle_out.close()

# for dirname, _, filenames in os.walk('./images/'):
#     for filename in filenames:
#         try:
#             for csv in csv_images:
#                 if csv == 'offensive_images':
#                     for off_img in csv_images['offensive_images']:
#                         if filename == off_img:
#                             data['label'].append(int(1))
#                 elif csv == 'not_offensive_images':
#                     for notOff_img in csv_images['not_offensive_images']:
#                         if filename == notOff_img:
#                             data['label'].append(int(0))
#
#                 data['image'].append(
#                     preprocess_image(
#                         os.path.join(dirname, filename)))
#
#                 data['filepath'].append(
#                     os.path.join(dirname, filename))
#
#                 data['text'].append(
#                     preprocess_txt(
#                         pytesseract.image_to_string(
#                             Image.open(
#                                 os.path.join(dirname, filename)))))
#
#         except Exception as e:
#             print(e)
#             continue
#
#         print('\r images: {} texts: {} labels : {}'.format(len(data['image']), len(data['text']), len(data['label'])),
#               end='')
#
# with open("dataa.pkl", "wb") as pickle_out:
#     pickle.dump(data, pickle_out)
# pickle_out.close()
