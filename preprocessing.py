import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import regex as re
import keras.utils as keras_image
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_image(filepath):
  img = keras_image.load_img(filepath , target_size=(224, 224), interpolation='bicubic')
  img = img.copy()
  img = keras_image.img_to_array(img)
  img = img//255.0

  return img

def preprocess_txt(text):
  word_Lemmatized = WordNetLemmatizer()
  text = text.lower()
  text = re.sub(r"\n"," ",text)
  text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
  text = re.sub(r'http\S+', '', text)
  stop = stopwords.words('english')
  pat = r'\b(?:{})\b'.format('|'.join(stop))
  text = text.replace(pat, '')
  text = text.replace(r'\s+', ' ')
  text = re.sub(r'[^a-zA-Z0-9 -]', '', text)
  text = re.sub('@[^\s]+','',text)
  text = word_tokenize(text)
  Final_words = []

  final_word = " ".join([word_Lemmatized.lemmatize(word) for word in text])
  Final_words.append(final_word)

  text = " ".join(Final_words)

  return text