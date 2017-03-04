import os
import pandas as pd
import gzip
import sys

script_dir = os.getcwd()
script_type = "MAC"

if script_type == "MAC":
    DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
    METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'
    os.chdir(script_dir)
elif script_dir == 'WIN':
    DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
    METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'
    os.chdir(script_dir)
else:
    DATA_PATH = os.path.join(script_dir, 'userSplit')
    METADATA_PATH = os.path.join(script_dir, 'metaSplit')
    os.chdir(script_dir)

REVIEW_COLS = ["reviewerID", 'asin', 'helpful', 'overall', 'review_length', 'summary_length']
METADATA_COLS = ['asin', 'price', 'brand']

# start = dt.datetime.now()

os.mkdir(os.getcwd(), 'processed_data')
pickle_path = os.path.join(os.getcwd(), 'processed_data')


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        try:
            yield eval(l)
        except:
            pass


def get_df(path, metadata=False):
    file_name_appender = path[-4:]
    i = 0
    df = {}
    for d in parse(path):
        print i
        try:
            if not metadata:  # Extract only the columns we are interested in to save memory
                d['review_length'] = len(d['reviewText'])
                d['summary_length'] = len(d['summary'])
                for keys in d.keys():
                    if keys not in REVIEW_COLS:
                        del d[keys]
                if 'helpful' in d.keys():
                    d['upvotes'] = d['helpful'][0]
                    try:
                        d['helpfulness'] = round(float(d['helpful'][0]) / float(d['helpful'][1]), 2)
                    except ZeroDivisionError:
                        d['helpfulness'] = 0.0
            else:
                for keys in d.keys():
                    if keys not in METADATA_COLS:
                        del d[keys]

            df[i] = d
        except:
            pass
    print "Storing Dataframe..."
    data_frame = pd.DataFrame.from_dict(df, orient='index')
    if not metadata:
        file_name = "user_data_" + str(file_name_appender) + ".p"
    else:
        file_name = "metadata_" + str(file_name_appender) + ".p"
    file_name_appender += 1
    data_frame.to_pickle(os.path.join(pickle_path, file_name))
    return  # pd.DataFrame.from_dict(df, orient='index')


# print "Creating Dataframes..."
# get_df(DATA_FILE)
# sys.exit()
# print "Dataframes created!"
print "Processing MetaData"
for metadata_file in os.listdir(METADATA_PATH):
    get_df(os.path.join(os.getcwd(), metadata_file))

print "Processing Review Data"
for reviewdata_file in os.listdir(DATA_PATH):
    get_df(os.path.join(os.getcwd(), reviewdata_file))