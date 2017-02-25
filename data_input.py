import os
import pandas as pd
import numpy as np
import gzip
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt4Agg')

# from matplotlib import style
# style.use("ggplot")
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import Counter


os.chdir('/Volumes/Expansion/Amazon_Review_Data')

REVIEW_COLS = ["reviewerID", 'asin', 'helpful', 'overall']
METADATA_COLS = ['asin', 'price', 'brand']

DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'


# kydef.org/AmazonDataset/

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_df(path, metadata=False):
    i = 0
    df = {}
    for d in parse(path):
        if not metadata:  # Extract only the columns we are interested in to save memory
            for keys in d.keys():
                if keys not in REVIEW_COLS:
                    del d[keys]
            if 'helpful' in d.keys():
                d['upvotes'] = d['helpful'][0]
                try:
                    d['helpfullness'] = round(float(d['helpful'][0]) / float(d['helpful'][1]), 2)
                except ZeroDivisionError:
                    d['helpfullness'] = 0.0
        else:
            for keys in d.keys():
                if keys not in METADATA_COLS:
                    del d[keys]

        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


print "Creating Dataframes..."
df = get_df(DATA_FILE)
print "Dataframes created!"

merged_df = None


# This method the the correct one.
def get_product_means(df):
    """
    Assume that we are taking all the products that were reviewed and finding the average
    number of reviews for that product compared to the total number of products
    :param df: The input dataframe containing all the review data
    :return: Mean review for each product
    """

    # The review counts for each product
    item_counts = df['asin'].value_counts()
    # print item_counts[:10]

    # Total number of reviews
    total_num_reviews = len(df.index)

    return item_counts/total_num_reviews


# For each item, show what the mean review was.


def get_mean_reviews(df):  # Incorrect method
    """
    Total number of reviews / Total number of items
    :param df:
    :return:
    """
    # print len(df.index)
    # print len(df['asin'].unique())
    return len(df.index) / len(df['asin'].unique())


def get_mode_reviewed(df):
    """
    Taking in a dataframe of reviews, return the one that reviewed the most
    :param df: The input dataframe containing all the review data
    :return Tuple containing itemid, number_of_reviews
    """
    item_counts = df['asin'].value_counts()
    # print item_counts
    review_count = item_counts[0]
    item = item_counts[item_counts == review_count].index[0]
    return item, review_count


def get_median_reviewed(df):
    """
    Get the item with the middle amount of reviews (in terms of number of reviews)
    :param df: Input dataframe containing all review data
    :return: Tuple containing itemid, number_of_reviews
    """
    item_counts = df['asin'].value_counts()
    median_item = item_counts.median()
    # print median_item
    item = item_counts[item_counts == median_item].index[0]
    # print item
    return item, median_item


def get_review_distribution(df):
    """
    Get the distribution of the reviews across all products
    :param df: Input dataframe containing all review data
    :return: Series containing the review number and the count of each
    """
    # TODO Research the definition of skewing and how it relates to this
    # review_counts = df['overall'].value_counts()
    # return review_counts
    grouped = df.groupby(by='asin')['overall'].mean()
    # print df['overall'].unique()
    grouped = grouped.to_frame()
    grouped['skew'] = np.where(grouped['overall'] > 3, 'positive', 'negative')
    grouped.ix[grouped.overall == 3, 'skew'] = 'symmetrical'
    grouped.columns = ['meanReviews', 'skew']
    print grouped[:10]
    grouped.to_csv('review_distribution.csv')


def get_user_reviews(df):

    user_reviews = df['reviewerID'].value_counts()
    max_user_review = user_reviews[0]
    user = user_reviews[user_reviews == max_user_review].index[0]
    return user, max_user_review


def get_most_helpful_reviewer(df):

    # Iterate over the dataframe and create a new column 'upvotes' based on
    # how much people found their review helpful
    # for i, helpful_text in enumerate(df['helpful']):
    #     df.ix[i, 'upvotes'] = helpful_text[0]

    # Group by the reviewerName and upvotes sum to create a Series
    grouped = df.groupby(by='reviewerID')['upvotes'].sum()

    # Get the entry that had the max number of upvotes then extract the user
    max_upvotes = max(grouped)
    # print max_upvotes
    user = grouped[grouped == max_upvotes].index[0]
    return user, max_upvotes


def get_most_and_least_expensive_high_review_product(df1):
    """
    Retrieves the prices of the most expensive and least expensive
    products that got a rating >= 4
    :param df: Dataframe containing review and metadata info
    :return: dict containing {most_exp_product: price, least_exp_product: price}
    """

    global merged_df
    # We use a different dataset to link the reviews and metadata. Original dataframe didn't have this info.
    # df1 = get_df('reviews_Amazon_Instant_Video.json.gz')
    df2 = get_df(METADATA_FILE, metadata=True)
    df3 = pd.merge(df1, df2, on='asin', how='outer')

    merged_df = df3  # To be used later in Kmeans
    product_filter = df3['overall'] >= 4.0
    high_reviewed_products = df3[product_filter]
    # print high_reviewed_products[:10]
    # The data contained NaN so we use the nanmax/min funtions to get max/min
    most_exp = round(np.nanmax(high_reviewed_products['price'])[0], 2)
    least_exp = round(np.nanmin(high_reviewed_products['price'])[0], 2)

    most_exp_prod = df3.loc[df3['price'] == most_exp, 'asin'].iloc[0]
    least_exp_prod = df3.loc[df3['price'] == least_exp, 'asin'].iloc[0]
    return {most_exp_prod: most_exp, least_exp_prod: least_exp}


def cluster_data(df):
    # filter out NaNs

    cluster_num = 5
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    # df = df[pandas.notnull(df['upvotes'])]
    # df = df[pandas.notnull(df['helpfullness'])]
    # df = df[pandas.notnull(df['overall'])]
    # df = df[pandas.notnull(df['price'])]

    kmeans = MiniBatchKMeans(n_clusters=cluster_num)
    kmeans.fit(df)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # The below code works but it takes a long time to render graph

    colors = ["g.", "r.", "c.", "y."]

    color = np.random.rand(cluster_num)
    c = Counter(labels)

    fig = plt.figure()
    ax = ax = fig.gca(projection='3d')

    for index, row in df.iterrows():
        ax.scatter(row['upvotes'], row['overall'], row['price'], c=color[labels[index]])
        if index == 10000:
            break

    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker="x", s=150, linewidths=5, zorder=100, c=color)
    plt.show()

# get_mean_reviews(df)
get_review_distribution(df)
# print df[:10]

# get_most_and_least_expensive_high_review_product(df)
# new_df = merged_df[['upvotes', 'overall', 'price']].copy()
# cluster_data(new_df)


# print get_most_helpful_reviewer(df)
# print list(merged_df.columns.values)
# print list(df)
# print list(df2)

# print get_product_means(df)[:10]




