import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


train_texts, test_texts, train_labels, test_labels = train_test_split(train['text'], train['target'], test_size=0.2, random_state=42)
