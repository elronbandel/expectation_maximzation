from collections import Counter, defaultdict
from itertools import groupby, chain
from operator import itemgetter


def load(file):
    with open(file) as f:
        res = []
        for line in f:
            if line.startswith('<T'):
                labels = line[1:-2].split()[2:]
            elif line != '\n':
                res.append((line.strip(), labels))
        return res


def filter_rares(counter, threshold):
    return Counter({k: v for k, v in counter.items() if v > threshold})


def filter_rare_words(data, threshold):
    overall_count = Counter(get_words(text for text, labels in data))
    filtered_count = filter_rares(overall_count, threshold=threshold)
    filtered = [(' '.join(word for word in text.split() if word in filtered_count), label) for text, label in data]
    return filtered, filtered_count


def init(texts):
    return [(text, i % 9) for i, text in enumerate(texts)]


def group_by_topic(data):
    groups = defaultdict(list)
    for text, topic in data:
        groups[topic].append(text)
    return groups.items()


def get_words(texts):
    for text in texts:
        for word in text.split():
            yield word


def count_word_by_topic(data):
    count = defaultdict(Counter)
    for topic, group in group_by_topic(data):
        count[topic] = Counter(get_words(group))
    return count


def count_words_in_topic(word_by_topic_count):
    return {topic: sum(word_count.values()) for topic, word_count in word_by_topic_count.items()}


def count_words_by_doc(data):
    return {i: Counter(doc.split()) for i, doc in enumerate(data)}


def count_docs(data):
    return len(data)


def count_docs_in_topic(data):
    return Counter(label for text, label in data)


# p(w_k|x_i):
def prob_word_by_topic(word, topic, word_by_topic_count, words_in_topic_count, vocab_size, l):
    return (l + word_by_topic_count[topic][word]) / (words_in_topic_count[topic] + l * vocab_size)


# p(x_i)
def prob_topic(topic, docs_in_topic_count, docs_count, topics_size, l):
    return (l + docs_in_topic_count[topic]) / (docs_count + topics_size * l)


def doc_by_topic_nominator(topic, doc, doc_idx, docs_in_topic_count, word_by_doc_count, docs_count, topics_size,
                           word_by_topic_count,
                           words_in_topic_count, vocab_size, l):
    topic_prob = prob_topic(topic, docs_in_topic_count, docs_count, topics_size, l)
    if topic_prob == 0:
        pass
    res = topic_prob
    for word in doc.split():
        word_by_topic_prob = prob_word_by_topic(word, topic, word_by_topic_count, words_in_topic_count, vocab_size, l)
        if word_by_topic_prob == 0:
            pass
        res_before = res
        temp_res = word_by_topic_prob ** word_by_doc_count[doc_idx][word]
        res *= temp_res
        if res == 0:
            pass
    return res


# doc_by_topic_nominator_dict = {'topic1':{doc1:0.3, doc2:0.2, 'topic2':...}
def doc_denominator(doc, doc_by_topic_nominator_dict):
    return sum(docs_nominator[doc] for topic, docs_nominator in doc_by_topic_nominator_dict.items())


def prob_topic_by_doc(topic, doc, doc_by_topic_nominator_dict, doc_denominator_dict):
    return doc_by_topic_nominator_dict[topic][doc] / doc_denominator_dict[doc]


def expectation(labeled_docs, vocab_size):
    docs = [doc for doc, label in labeled_docs]
    n_docs = count_docs(labeled_docs)

    word_by_doc_count = count_words_by_doc(docs)
    docs_in_topic_count = count_docs_in_topic(labeled_docs)
    word_by_topic_count = count_word_by_topic(labeled_docs)
    words_in_topic_count = count_words_in_topic(word_by_topic_count)

    topics = docs_in_topic_count.keys()
    n_topics = len(topics)

    l = 0.001

    doc_by_topic_nominator_dict = defaultdict(Counter)
    for topic in topics:
        for i, doc in enumerate(docs):
            doc_by_topic_nominator_dict[topic][i] = doc_by_topic_nominator(topic, doc, i, docs_in_topic_count,
                                                                             word_by_doc_count, n_docs, n_topics,
                                                                             word_by_topic_count,
                                                                             words_in_topic_count, vocab_size, l)
    doc_denominator_dict = Counter()
    for doc_idx in range(len(docs)):
        doc_denominator_dict[doc_idx] = doc_denominator(doc_idx, doc_by_topic_nominator_dict)

    res = defaultdict(Counter)
    for topic in topics:
        for doc_idx in range(len(docs)):
            res[topic][doc_idx] = prob_topic_by_doc(topic, doc_idx, doc_by_topic_nominator_dict, doc_denominator_dict)
    return res


def main():
    dev = load('dataset/develop.txt')
    dev, dev_count = filter_rare_words(dev, threshold=3)
    vocab_size = len(dev_count.keys())
    initial = init(text for text, labels in dev)
    w = expectation(initial, vocab_size)
    print(w)


if __name__ == '__main__':
    main()
