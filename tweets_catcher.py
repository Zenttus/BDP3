from twitter import OAuth, TwitterStream
from hdfs_manager import HDFSManager
import config
import subprocess
import time
from time import strftime, gmtime

current_milli_time = lambda: int(round(time.time() * 1000))


def get_tweets(hdfsmanager):
    '''
    It starts the recolection of tweets. And saves them using an HDFSManager.
    :param hdfsmanager: Initiated HDFSManager
    :return: None
    '''
    assert hdfsmanager.__class__ == HDFSManager

    authorization = OAuth(config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET, config.CONSUMER_KEY, config.CONSUMER_SECRET)

    stream = TwitterStream(auth=authorization)

    tweets = stream.statuses.sample()

    for tweet in tweets:
        try:
            if len(tweet) > 2:  #This filters the "deleted" tweets
                hdfsmanager.save_tweet(tweet)
        except Exception as e:
            print(e)
            pass


class HDFSManager:

    def __init__(self):

        # Create folder path
        subprocess.Popen(['hdfs dfs -mkdir ' + config.OUTPUT_FILE_PATH], shell=True)

        # Start tracking time
        self.tick = current_milli_time()

        # Define file path
        self.currentfile = config.OUTPUT_FILE_PATH + strftime("%d%b%Y_%H%M%S", gmtime()) + ".json"

    def update_file_list(self):

        # Creates list to keep track of files
        filelist = open(config.LIST_PATH + "./tweetsList.txt", "a+")
        filelist.write(self.currentfile + "\n")
        filelist.close()

        # Define new file path
        self.currentfile = config.OUTPUT_FILE_PATH + strftime("%d%b%Y_%H%M%S", gmtime()) + ".json"

        # Restart countdown
        self.tick = current_milli_time()

    def save_tweet(self, tweet):

        try:
            output = open("./temp.json", "a+", encoding='utf-8')
            output.write(str(tweet) + '\n')
        except Exception as e:
            print(e)
        finally:
            output.close()

        #Once the interval is done, send file to HDFS and start a new one.

        if current_milli_time() - self.tick > config.INTERVAL * 1000:
            print("Moving tweets to hdfs...")
            # Send new file to HDFS
            put = subprocess.Popen(['hdfs dfs -put ./temp.json ' + self.currentfile], shell=True)
            put.communicate()

            self.update_file_list()

            # Clear temp file
            open("./temp.json", "w").close()
            print("DONE")

