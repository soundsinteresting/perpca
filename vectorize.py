import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json



USUAL_WORDS = [ '','and','And','but','But','uh','or','if','because','so','So','also',
               'of','for','to','in','on','out','at','with','about','as','into','by','from',
               'the','The','that','thats','Thats','this','a','an','all','too','some','very',
               'is','are','be','have','has','had','not','been','Ive',
               'what','when','who','how','What','When','which',
               'was','were','do','does','did','will','would','should','can','Ill','going','dont',
               'want','get','got','think','every','also','said',
               'Mr',
               'I','Im','my','me','he','He','his','her','she','She',
               'we','We','us','our','you','You','your','They','they','them','Them','their','it','It','its','Its',
               'D','000']

def create_vector(word_count, vocab_size, wn):
    vec = np.zeros(vocab_size)
    for word in word_count:
        if word in wn.keys():
            vec[wn[word]] = word_count[word]
    return vec


def top_words(vector, number2word, top=5):
    positions = np.argpartition(np.abs(vector), -top)[-top:]
    words = []
    for position in positions:
        words.append(number2word[position])
    return words

#if __name__ == '__main__':
def vectorize_words():
    source_dir = r"debate"  # r"./organized_results/train_output"
    filenames = [os.path.join(source_dir, name) for name in os.listdir(source_dir)]
    all_dialogue = []
    word2number = dict()
    word2count = dict()

    total_words = 0
    years = []
    for name in filenames:
        years.append(int(name[-9:-5]))
        file = open(name)
        data = json.load(file)

        one_debate = []
        for speech in data['content']:
            dialogue = speech['dialogue']
            #import re
            words = re.split('\W+', dialogue)

            speech_dict = dict()
            for word in words:
                if not word in USUAL_WORDS:
                    #if not word in word2number.keys():
                    #    word2number[word] = len(word2number)
                    if not word in word2count.keys():
                        word2count[word] = 0
                    word2count[word] += 1
                    if not word in speech_dict.keys():
                        speech_dict[word] = 0
                    speech_dict[word] += 1
                    total_words += 1
            if len(speech_dict)>0:
                one_debate.append(speech_dict)
        all_dialogue.append(one_debate)
    #print(years)
    #print(word2number)

    #print(all_dialogue)


    #print(word2count.values())
    #wc = np.sort(np.array(word2count.values()))
    wc = {k: v for k, v in sorted(word2count.items(), key=lambda item: item[1], reverse=True) if v > 5}
    print("total words %d, total dictionary %d" % (total_words, len(wc)))
    print("total debates %d" % len(all_dialogue))
    word2number = {w:i for i,w in enumerate(wc.keys())}
    number2word = {i:w for i,w in enumerate(wc.keys())}

    vocab_size= len(word2number)

    #print(wc)
    allyears = sorted(set(years))
    #for
    #print(years)
    #print(allyears)
    all_data = []
    for year in allyears:
        data = []
        for id, diags in enumerate(all_dialogue):
            if not years[id] == year:
                continue
            num_diag = len(diags)
            debate = np.zeros((num_diag, vocab_size))
            for i,speech in enumerate(diags):
                vec = create_vector(speech, vocab_size, word2number)
                debate[i] += vec
            data.append(debate)
        data = np.concatenate(data)
        all_data.append(data)
    return all_data, number2word, allyears

    '''
    smat = np.concatenate(data)
    print(len(smat))

    print('concatenated, calculating SVD')
    u, s, vh = np.linalg.svd(smat)
    print('SVD calculated')
    for j in range(5):
        print('top %d principal component: '%j)
        words = top_words(vh[j], number2word, top=40)
        print(words)
        print('======================')
    '''

if __name__ == "__main__":
    all_data = vectorize_words()
    for data in all_data:
        print(data.shape)






