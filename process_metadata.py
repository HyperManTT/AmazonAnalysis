import os
import gzip
import pandas as pd

script_dir = os.getcwd()
script_type = "SERVER"

if script_type == "MAC":
    DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
    METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'
    os.chdir(script_dir)
elif script_dir == 'WIN':
    DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
    METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'
    os.chdir(script_dir)
else:
    DATA_FILE = 'aggressive_dedup.json.gz'
    METADATA_FILE = 'metadata.json.gz'
    os.chdir(script_dir)


REVIEW_COLS = ["reviewerID", 'asin', 'helpful', 'overall', 'review_length', 'summary_length']
METADATA_COLS = ['asin', 'price', 'brand']

DATA_PICKLE_FILENAME = 'reviews.p'
METADATA_PICKLE_FILENAME = 'metadata.p'


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        try:
            yield eval(l)
        except:
            pass


def get_df(path, metadata=False):
    i = 0
    df = {}
    for d in parse(path):
        if i % 100000 == 0:
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
            i += 1
        except:
            pass
    return pd.DataFrame.from_dict(df, orient='index')


try:
    if not os.path.exists(os.path.join(os.getcwd(), METADATA_PICKLE_FILENAME)):
        print "Creating Metadata Dataframe..."
        df2 = get_df(METADATA_FILE, metadata=True)
        df2.to_pickle(METADATA_PICKLE_FILENAME)
        print "MetaData Dataframes created!"
    else:
        print "Reading pickled Metadata Dataframe"
        df2 = pd.read_pickle(os.path.join(os.getcwd(), METADATA_PICKLE_FILENAME))
except Exception as e:
    print str(e)