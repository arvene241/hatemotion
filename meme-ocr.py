import cv2
import pytesseract
import pandas as pd
import os
from preprocessing import preprocess_image, preprocess_txt
import pickle

df = pd.read_csv("test_dataset2.csv")

try:
    from PIL import Image
except ImportError:
    import Image

data = {"image": [], "filepath": [], "text": [], "label": []}

# Iterate over the rows in the DataFrame
for index, row in df.iterrows():
    # Get the label and image values for the current row
    label = row['label']
    image = row['image_name']
    # Check the label value and append the image to the corresponding list
    try:
        if label == 'offensive':
            data['label'].append(int(1))
        if label == 'not_offensive':
            data['label'].append(int(0))

        data['image'].append(
            preprocess_image(
                os.path.join('test images/', image)))

        data['filepath'].append(
            os.path.join('test images/', image))

        data['text'].append(
            preprocess_txt(
                pytesseract.image_to_string(
                    Image.open(
                        os.path.join('test images/', image)))))

    except Exception as e:
        print(e)
        continue

    print('\r images: {} texts: {} labels : {}'.format(len(data['image']), len(data['text']), len(data['label'])), end='')

# for text in data['text']:
#     print(text)
#     df['actual_cleanded_text'] = text
#     print(df['actual_cleanded_text'])
#     df.to_csv('new.csv', index=False)
df = pd.DataFrame.from_dict(data)
df.to_csv('new1.csv', index=False)

with open("data.pkl", "wb") as pickle_out:
    pickle.dump(data, pickle_out)
pickle_out.close()
