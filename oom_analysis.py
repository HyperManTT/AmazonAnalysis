import pandas as pd
import os
import gzip
import datetime as dt
from sqlalchemy import create_engine


REVIEW_COLS = ["reviewerID", 'asin', 'overall', 'review_length', 'summary_length', 'upvotes', 'helpfulness']
METADATA_COLS = ['asin', 'price']

script_dir = os.getcwd()
script_type = "MAC"

if script_type == "MAC":
    DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
    METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'
    # os.chdir('/Volumes/Expansion/Amazon_Review_Data')
elif script_type == 'WIN':
    DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
    METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'
    os.chdir(script_dir)
else:
    DATA_FILE = 'aggressive_dedup.json.gz'
    METADATA_FILE = 'metadata.json.gz'
    os.chdir(script_dir)

start = dt.datetime.now()
disk_engine = create_engine('sqlite:///amazon.db')


def parse(path):
    # print path
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_df(path, metadata=False):
    i = 0
    index_start = 1
    for d in parse(path):
        df = {}
        if not metadata:  # Extract only the columns we are interested in to save memory
            d['review_length'] = len(d['reviewText'])
            d['summary_length'] = len(d['summary'])
            if 'helpful' in d.keys():
                d['upvotes'] = d['helpful'][0]
                try:
                    d['helpfulness'] = round(float(d['helpful'][0]) / float(d['helpful'][1]), 2)
                except ZeroDivisionError:
                    d['helpfulness'] = 0.0
            for keys in d.keys():
                if keys not in REVIEW_COLS:
                    del d[keys]
        else:
            if 'price' not in d.keys():
                continue
            for keys in d.keys():
                if keys not in METADATA_COLS:
                    del d[keys]

        df[i] = d
        # print df
        try:
            # print df
            df = pd.DataFrame.from_dict(df, orient='index')
            # df.index = [index_start]
            # print df.index
            if not metadata:
                df.columns = REVIEW_COLS
                df.to_sql('reviews', disk_engine, if_exists='append')
            else:
                # print df
                df.columns = METADATA_COLS
                df.to_sql('meta', disk_engine, if_exists='append')
            index_start += 1
            # print index_start
        except Exception as e:
            # print df
            print str(e)
            pass
        print '{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, i)
        i += 1
    return  # pd.DataFrame.from_dict(df, orient='index')


def parse_data(path, metadata=False):
    chunksize = 20000
    j = 0
    index_start = 1

    for df in pd.read_json(path, chunksize=chunksize, iterator=True, encoding='utf-8'):

        df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})  # Remove spaces from columns

        df.index += index_start

        # Remove the un-interesting columns
        if not metadata:
            columns = ["reviewerID", 'asin', 'helpful', 'overall', 'review_length', 'summary_length']
            dbname = 'reviews'
        else:
            columns = ['asin', 'price', 'brand']
            dbname = 'meta'
        for c in df.columns:
            if c not in columns:
                df = df.drop(c, axis=1)

        j += 1
        print '{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j * chunksize)

        df.to_sql(dbname, disk_engine, if_exists='append')
        index_start = df.index[-1] + 1


if not os.path.exists(os.path.join(os.getcwd(), 'amazon.db')):
    print "Creating Dataframes..."
    get_df(DATA_FILE)
    get_df(METADATA_FILE, metadata=True)
    # parse_data(METADATA_FILE, metadata=True)
    print "Dataframes created!"
    # print sum(1 for i in parse(DATA_FILE))

# df = pd.read_sql_query('SELECT count(*) FROM meta', disk_engine)
# print df.head()
