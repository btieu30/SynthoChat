#testing natural language processing
import nltk
from nltk import word_tokenize, WordNetLemmatizer, pos_tag, ne_chunk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


#process the text to get the basic gist / important words
def preprocess(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word not in stop_words]
    return words

def find_locations(text):
    # Tokenize the text into sentences
    sens = nltk.sent_tokenize(text)
    locations = []

    for sen in sens:
        # Tokenize each sentence into words
        words = word_tokenize(sen)
        # POS tagging
        tags = pos_tag(words)
        # Named Entity Recognition
        location = ne_chunk(tags, binary=False)

        for chunk in location:
            if hasattr(chunk, 'label') and chunk.label() == 'GPE':
                    city = ' '.join(c[0] for c in chunk)
                    locations.append(city)

    return locations

text = "weather at 5 in New York and London"
important_words = preprocess(text)
print(preprocess(text))

if 'weather' in important_words:
    location = find_locations(text)
    if location:
        print('Weather location:', location)
    else:
        print('Please provide a city name for the weather')