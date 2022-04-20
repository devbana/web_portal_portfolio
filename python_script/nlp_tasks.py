from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle as pk
import re
import spacy


class nlp_chacha:

	def __init__(self):
		self.nlp_hotel = spacy.load('en_core_web_md')

	def text_preprocessing(self, document):
		result = []
		document_n = str(document).lower()  # Lowering the document
		document_n_ = re.sub('[^a-z0-9]', ' ', document_n)  # Substitute the non alpha numeric to space
		for i in self.nlp_hotel(document_n_):
			if i.is_digit is True or i.is_stop is True:
				pass
			else:
				if i.pos_ is 'SPACE':
					pass
				else:
					result.append(str(i.lemma_))
		return ' '.join(result)

	def hotel_reviews(self, path_, process_rev):
		tfidf = pk.load(open(path_ + '/aimodels/hotel_reviews/tfidf_vectorizer.pickle', 'rb'))
		lsa_model = pk.load(open(path_ + '/aimodels/hotel_reviews/lsa_model.pickle', 'rb'))
		vect = tfidf.transform([process_rev])

		predicted = {}
		for i, j in enumerate(lsa_model.transform(vect)[0]):
			predicted[i] = j
		sortedd = []
		for i in sorted(predicted.values(), reverse=True):
			for l, k in predicted.items():
				if k > 0 and k == i:
					sortedd.append(l)
		return sortedd

	def hotel_prediction(self, path_, document_):
		tfidf = pk.load(open(path_ + '/aimodels/hotel_reviews/model_tokenizer.pickle', 'rb'))
		vect = tfidf.texts_to_sequences([document_])
		text_sequences = pad_sequences(vect, maxlen=50)
		hotel_model = load_model(path_ + '/aimodels/hotel_reviews/hotel_keras_99.h5')
		result = hotel_model.predict(text_sequences)
		return result
