import pandas as pd
import os
import gzip
import datetime as dt
from sqlalchemy import create_engine


REVIEW_COLS = ["reviewerID", 'asin', 'helpful', 'overall', 'review_length', 'summary_length']
METADATA_COLS = ['asin', 'price', 'brand']
chunksize = 20000
j = 0


script_dir = os.getcwd()
script_type = "SERVER"

if script_type == "MAC":
    DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
    METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'
    os.chdir('/Volumes/Expansion/Amazon_Review_Data')
elif script_type == 'WIN':
    DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
    METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'
    os.chdir(script_dir)
else:
    DATA_FILE = 'aggressive_dedup.json.gz'
    METADATA_FILE = 'metadata.json.gz'
    os.chdir(script_dir)

start = dt.datetime.now()


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_df(path, metadata=False):
    i = 0
    df = {}
    for d in parse(path):
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
        print '{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, i)
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

print "Creating Dataframes..."
df = get_df(DATA_FILE)
print "Dataframes created!"
# print sum(1 for i in parse(DATA_FILE))