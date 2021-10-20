'''
Visualize Topics
'''

if __name__ ==  '__main__':
    import argparse
    from gensim.test.utils import common_texts
    from utils import *
    from contextualized_topic_models.evaluation.measures import TopicDiversity, CoherenceNPMI, InvertedRBO
    import csv
    from utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', default='SeaNMF/data/doc_term_mat.txt', help='term document matrix file')
    parser.add_argument('--vocab_file', default='SeaNMF/data/vocab.txt', help='vocab file')
    parser.add_argument('--par_file', default='SeaNMF/seanmf_results/60/W.txt', help='model results file')
    opt = parser.parse_args()

    docs = read_docs(opt.corpus_file)
    vocab = read_vocab(opt.vocab_file)
    n_docs = len(docs)
    n_terms = len(vocab)
    print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

    dt_mat = np.zeros([n_terms, n_terms])
    for itm in docs:
        for kk in itm:
            for jj in itm:
                if kk != jj:
                    dt_mat[int(kk), int(jj)] += 1.0
    print('co-occur done')
            
    W = np.loadtxt(opt.par_file, dtype=float)
    n_topic = W.shape[1]
    print('n_topic={}'.format(n_topic))

    PMI_arr = []
    n_topKeyword = 10
    for k in range(n_topic):
        topKeywordsIndex = W[:,k].argsort()[::-1][:n_topKeyword]
        PMI_arr.append(calculate_PMI(dt_mat, topKeywordsIndex))
    print('Average PMI={}'.format(np.average(np.array(PMI_arr))))

    index = np.argsort(PMI_arr)
    
    #list for all topics
    all_topics = []

    for k in index:
        #list for the current topic words
        topic_words = []
        print('Topic ' + str(k+1) + ': ', end=' ')
        print(round(PMI_arr[k],3), end=' ')
        for w in np.argsort(W[:,k])[::-1][:n_topKeyword]:
            print(vocab[w], end=' ')
            topic_words.append(vocab[w])
        all_topics.append(topic_words)
        print()

    with open("dataset/topic_word_lists.csv","w") as f:
        wr = csv.writer(f)
        wr.writerows(all_topics)

    ##### printing the different score values
    #Get the topic diversity
    td = TopicDiversity(all_topics)
    print("Topic diversity score is:", td.score(topk=10))

    #Get the inverted RBO
    rbo = InvertedRBO(all_topics)
    print("Inverted RBO score is:", rbo.score())

    # Get a measure of the NPMI
    with open('/Users/borismarinov/Desktop/Medium/3d-networks/SeaNMF/data/topic_modeling.txt',"r") as fr:
        texts = [doc.split() for doc in fr.read().splitlines()]

    npmi = CoherenceNPMI(texts=texts, topics=all_topics)
    print("NPMI SCORE ON {0} (ALL) TEXTS IS:".format(len(texts)), npmi.score())
