#import tweets_catcher
#from tweets_catcher import HDFSManager
import tweets_catcher
from tweets_catcher import HDFSManager
import visualizer

from text_classifiers import TextClassifierModel1, TextClassifierModel2

if __name__ == '__main__':

    visualizer.run()
    #manager = HDFSManager()
    #tweets_catcher.get_tweets(manager)
