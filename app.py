from flask import Flask, render_template, request, jsonify
import json
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
from nltk.chunk import ne_chunk

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    uploaded_file = request.files['file']

    if uploaded_file:
        text = uploaded_file.read().decode('utf-8')

    results = {}

    results['morphological'] = morphological(text)
    results['lexical'] = lexical(text)
    results['syntax'] = syntax(text)

    json_data = json.dumps(results, indent=4)

    return jsonify(results)


def morphological(sent):
    tokens = word_tokenize(sent)
    tagged_tokens = pos_tag(tokens)
    morph = []
    for token, pos in tagged_tokens:
        morph.append((token, pos))
    return morph


def lexical(sent):
    tokens = word_tokenize(sent)
    lexi = []
    for token in tokens:
        lexi.append(("Token:", token))
    return lexi


def syntax(sent):
    tokens = word_tokenize(sent)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    pos_tags = pos_tag(tokens)
    stop_words = get_stop_words('en')
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return list(zip(tokens, lemmas, [tag for _, tag in pos_tags], [tag for _, tag in pos_tags], filtered_tokens))


def semantic_entity(text: str):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)
    sem_en = []
    for entity in named_entities.subtrees():
        if entity.label() != 'S':
            sem_en.append((' '.join(word for word, pos in entity.leaves()), entity.label()))
    return sem_en


def semantic_similar_words(sentences):
    # Split sentences into words
    sentences = sentences.split("\n")
    sentences = [word_tokenize(sentence) for sentence in sentences]

    # Remove stopwords and perform part-of-speech tagging
    stop_words = get_stop_words('en')
    tagged_sentences = []
    for sentence in sentences:
        tagged_sentence = pos_tag(sentence)
        filtered_sentence = [(word, tag) for word, tag in tagged_sentence if word.lower() not in stop_words]
        tagged_sentences.append(filtered_sentence)

    # Create WordNet similarity matrix
    similarity_matrix = []
    for tagged_sentence in tagged_sentences:
        synsets = []
        for word, tag in tagged_sentence:
            if tag.startswith('N'):
                synset = wordnet.synsets(word, pos=wordnet.NOUN)
            elif tag.startswith('V'):
                synset = wordnet.synsets(word, pos=wordnet.VERB)
            elif tag.startswith('J'):
                synset = wordnet.synsets(word, pos=wordnet.ADJ)
            elif tag.startswith('R'):
                synset = wordnet.synsets(word, pos=wordnet.ADV)
            else:
                synset = []
            synsets.extend(synset)
        similarity_matrix.append(synsets)

    # Calculate most similar words
    similar_words = {}
    for i, synsets in enumerate(similarity_matrix):
        fdist = FreqDist()
        for synset in synsets:
            for lemma in synset.lemmas():
                fdist[lemma.name()] += 1
        similar_words[sentences[i]] = fdist.most_common(10)

    return similar_words


def pragmatic(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Perform part-of-speech tagging
    tagged_tokens = pos_tag(tokens)

    # Apply named entity recognition (NER)
    ner_tags = ne_chunk(tagged_tokens)

    prag = []
    # Analyze the pragmatics of named entities
    for subtree in ner_tags.subtrees():
        if subtree.label() == 'GPE':  # Filter for geographical entities
            entity = ' '.join(word for word, tag in subtree.leaves())
            context = ' '.join(word for word, _ in subtree.sent)
            prag.append({
                "Entity": entity,
                "Label": subtree.label(),
                "Context": context
            })

    return prag


@app.route('/download')
def download():
    analysis_type = request.args.get('analysis')
    if analysis_type == 'morphological':
        data = {"result": "This is the morphological analysis result."}
        filename = 'morphological_analysis.json'
    elif analysis_type == 'lexical':
        data = {"result": "This is the lexical analysis result."}
        filename = 'lexical_analysis.json'
    elif analysis_type == 'syntax':
        data = {"result": "This is the syntax analysis result."}
        filename = 'syntax_analysis.json'
    elif analysis_type == 'semantic_entity':
        data = {"result": "This is the syntax analysis result."}
        filename = 'semantic_entity_analysis.json'
    elif analysis_type == 'semantic_similar_words':
        data = {"result": "This is the syntax analysis result."}
        filename = 'semantic_similar_words_analysis.json'
    elif analysis_type == 'pragmatic':
        data = {"result": "This is the syntax analysis result."}
        filename = 'pragmatic_analysis.json'
    else:
        return 'Invalid analysis type'

    response = jsonify(data)
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


if __name__ == '__main__':
    app.run()
