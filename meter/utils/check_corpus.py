"""
Checks a corpus created from a Wikipedia dump file.
"""

import sys, time

def check_corpus(input_file):
    
    """Reads some lines of corpus from text file"""

    while(1):
        for lines in range(50):
            print(input_file.readline())
        user_input = input('>>> Type \'STOP\' to quit or hit Enter key for more <<< ')
        if user_input == 'STOP':
            break


def load_corpus(input_file):

    """Loads corpus from text file"""

    print('Loading corpus...')
    time1 = time.time()
    corpus = input_file.read()
    time2 = time.time()
    total_time = time2-time1
    print('It took %0.3f seconds to load corpus' %total_time)
    return corpus


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python check_wiki_corpus.py <corpus_file>')
        sys.exit(1)

    corpus_file = open(sys.argv[1],'r')
    check_corpus(corpus_file)
    corpus = load_corpus(corpus_file)