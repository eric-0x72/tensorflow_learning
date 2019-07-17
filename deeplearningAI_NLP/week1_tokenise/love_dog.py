from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
    'what are you doing?',
    'nothing'
]

tokenizer = Tokenizer(num_words=3)  # only keep most common n-1 words
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print(word_index)
print(tokenizer.texts_to_sequences(sentences))
