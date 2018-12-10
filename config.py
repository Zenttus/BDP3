# Twitter Authorization data
ACCESS_TOKEN = "" # Access token
ACCESS_TOKEN_SECRET = "" # Access token secret
CONSUMER_KEY = "" # API Key
CONSUMER_SECRET = "" # API secret key
# Output variables
HDFS_SERVER = 'http://0.0.0.0:9870' # Not used, the script has to be run on master.
OUTPUT_FILE_PATH = "/user/hdfs/tweets/" # HDFS path for tweets
LIST_PATH = "./" # Path to put file tracking tweets segments
#
KEYWORDS = ["Trump", "Flu", "Zika", "Diarrhea", "Ebola", "Headache", "Measles"]
INTERVAL = 60  # Sample size frequency in seconds
