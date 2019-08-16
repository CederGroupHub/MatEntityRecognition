__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# common variables
common_topics = [195, 193, 7, 175, 142, 139, 73, 54, 179, 76, 171, 66, 32, 59, 63]

def normalize_topic(topic):
    normalized_topic = []
    total_score = sum([topic[t] for t in topic])
    for t in topic:
        normalized_topic.append((t, topic[t]/total_score))
    normalized_topic = sorted(normalized_topic, key=lambda tmp_item: tmp_item[1], reverse=True)
    return normalized_topic

def get_sent_topic(sentence):
    """
    Get the topic distribution of a sentence.
    
    :param sentence: plain sentence text
    :return topic: dict of (topic, value) pairs
    """
    from synthesis_api_hub import Client

    with Client('synthesisproject.lbl.gov', 8005, 'topic') as topic_client:
        results = topic_client.infer_topics([sentence], 'lightlda_r0_sentence_topic_200')
    return normalize_topic(results[0][0])

def get_topics(str_words):
    """
    Get the key words that have a direct or indirect dependency relation with a material word.
    
    :param str_words: list of words, which froms a sentence
    :return topics: list of topic dicts
    """
    # goal
    topics = []
    for i in range(len(str_words)):
        topics.append([0.0]*len(common_topics))

    # reshape the words and add index
    sent_text = ' '.join(str_words)
    sent_topic = dict(get_sent_topic(sent_text))  
    sent_topic = ['{0:.3f}'.format(sent_topic.get(t, 0.0)) for t in common_topics]  
    for (tmp_index, word) in enumerate(str_words):
        if word == '<MAT>':
            topics[tmp_index] = sent_topic
    return topics

if __name__ == '__main__':
    print(get_topics(['The', '<MAT>', 'is', 'prepared', 'with', 'solid', 'state', 'synthesis', '.']))
    

