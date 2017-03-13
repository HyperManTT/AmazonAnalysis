import os
import pandas as pd
import gzip


"""
SCRIPT IS USED TO PROCESS THE REVIEW  DATAFILE

The output files are stored in a directory called 'processed_data'
"""

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

# start = dt.datetime.now()

if not os.path.exists(os.path.join(os.getcwd(), 'processed_data')):
    os.mkdir(os.path.join(os.getcwd(), 'processed_data'))
pickle_path = os.path.join(os.getcwd(), 'processed_data')


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        try:
            yield eval(l)
        except:
            pass


def get_df(path, metadata=False):
    file_name_appender = 0
    i = 1
    df = {}
    for d in parse(path):
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
            if i % 2000000 == 0:
                print "Pickling Dataframe " + str(file_name_appender) + "(" + str(i) + ")"
                data_frame = pd.DataFrame.from_dict(df, orient='index')
                if not metadata:
                    file_name = "user_data_" + str(file_name_appender) + ".p"
                else:
                    file_name = "metadata_" + str(file_name_appender) + ".p"
                file_name_appender += 1
                data_frame.to_pickle(os.path.join(pickle_path, file_name))
                df.clear()
        except:
            pass
        finally:
            i += 1
    return pd.DataFrame.from_dict(df, orient='index'), file_name_appender


# print "Creating Dataframes..."
# get_df(DATA_FILE)
# sys.exit()
# print "Dataframes created!"

if __name__ == "__main__":
    print "Processing Review Data"
    last_df, file_name_appender = get_df(os.path.join(os.getcwd(), DATA_FILE))
    file_name = "user_data_" + str(file_name_appender) + ".p"
    last_df.to_pickle(os.path.join(pickle_path, file_name))





