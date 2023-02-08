import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
import regex as re
from tensorflow import keras
import keras.utils as keras_image
import string

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
  tag_map = defaultdict(lambda: wn.NOUN)
  tag_map['J'] = wn.ADJ
  tag_map['V'] = wn.VERB
  tag_map['R'] = wn.ADV
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
  for word, tag in pos_tag(text):
      word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
      Final_words.append(word_Final)
  text = " ".join(Final_words)

  return text

# def preprocess_txt(text):
#   word_Lemmatized = WordNetLemmatizer()
#   text = str(text)
#
#   no_punct = "".join([c for c in text if c not in string.punctuation])
#   url_re_1 = r"\b(?:https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+(?:[/?].*)?"  # remove a maioria das urls
#   url_re_2 = r"(w{3}\.)*[a-zA-Z0-9]+\.{1}(co){1}[m]{0,1}\s{0,1}"  # remove any.com urls
#   url_re_3 = r"(w{3}\.)*[a-zA-Z0-9]+\.{1}(net){1}\s{0,1}"  # remove any.net urls
#   digits = r'[0-9]'  # remove digitos
#   excess_whitespace_removed = ' '.join(text.split())
#
#   s1 = re.sub(url_re_1, "", excess_whitespace_removed)
#   s2 = re.sub(url_re_2, "", s1)
#   s3 = re.sub(url_re_3, "", s2)
#   s4 = re.sub(digits, "", s3)
#   text = s4.translate(str.maketrans('', '', string.punctuation))
#
#   text = text.lower()
#   stop = [w for w in text if w not in stopwords.words('english')]
#   pat = r'\b(?:{})\b'.format('|'.join(stop))
#   text = text.replace(pat, '')
#   text = text.replace(r'\s+', ' ')
#
#   text = word_tokenize(text)
#
#   Final_words = []
#   final_word = " ".join([word_Lemmatized.lemmatize(word) for word in text])
#   Final_words.append(final_word)
#
#   text = " ".join(Final_words)
#
#   return text