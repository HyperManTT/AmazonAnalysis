import os
import pandas as pd
import numpy as np
import gzip
import sys
import matplotlib.pyplot as plt

from matplotlib import style
style.use("ggplot")
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


script_dir = os.getcwd()
script_type = "TEST"

if script_type == "TEST":
    DATA_FILE = 'reviews_Amazon_Instant_Video.json.gz'
    METADATA_FILE = 'meta_Amazon_Instant_Video.json.gz'
    os.chdir('/Volumes/Expansion/Amazon_Review_Data')
else:
    DATA_FILE = 'aggressive_dedup.json.gz'
    METADATA_FILE = 'metadata.json.gz'
    os.chdir(script_dir)


REVIEW_COLS = ["reviewerID", 'asin', 'helpful', 'overall', 'review_length', 'summary_length']
METADATA_COLS = ['asin', 'price', 'brand']


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
    return pd.DataFrame.from_dict(df, orient='index')


print "Creating Dataframes..."
df = get_df(DATA_FILE)
print "Dataframes created!"

merged_df = None


def write_df_tocsv(df, file_name):
    df.to_csv(os.path.join(script_dir, file_name))


def write_text_tofile(text):
    with open(os.path.join(script_dir, 'output_file.txt'), 'a') as output:
        output.write(text + '\n')


# This method is the correct one.
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

    mean_dataframe = item_counts/total_num_reviews
    write_df_tocsv(mean_dataframe, 'product_means.csv')
    return mean_dataframe


# For each item, show what the mean review was.


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
    write_text_tofile("Mode Reviewed: " + str(item) + ", Count: " + str(review_count))
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
    write_text_tofile("Median Reviewed: " + str(item) + ", Median: " + str(median_item))
    return item, median_item


def get_review_distribution(df):
    """
    Get the distribution of the reviews across all products
    :param df: Input dataframe containing all review data
    :return: Series containing the review number and the count of each
    """
    # TODO Research the definition of skewing and how it relates to this
    review_counts = df['overall'].value_counts()
    grouped = df.groupby(by='asin')['overall'].mean()
    grouped2 = df.groupby(['asin'])['overall'].agg(lambda x: x.value_counts().index[0])
    merged_group = pd.concat([grouped, grouped2], axis=1)
    merged_group.columns = ['mean', 'mode']
    merged_group['skew'] = np.where(merged_group['mean'] > merged_group['mode'], 'positive', 'negative')
    merged_group.ix[merged_group['mean'] == merged_group['mode'], 'skew'] = 'symmetrical'

    write_df_tocsv(merged_group, 'review_distribution.csv')

    # grouped3 = df.groupby(by='asin')['overall'].std()

    # grouped = grouped.to_frame()
    # grouped['skew'] = np.where(grouped['overall'] > 3, 'positive', 'negative')
    # grouped.ix[grouped.overall == 3, 'skew'] = 'symmetrical'
    # grouped.columns = ['meanReviews', 'skew']
    #print grouped
    # print grouped2
    # print grouped3

    # write_df_tocsv(grouped, 'review_distribution.csv')
    # grouped.to_csv('review_distribution.csv')


def get_user_reviews(df):

    user_reviews = df['reviewerID'].value_counts()
    max_user_review = user_reviews[0]
    user = user_reviews[user_reviews == max_user_review].index[0]
    write_text_tofile("Max User Review: " + str(user) + ", Review Count: " + str(max_user_review))
    return user, max_user_review


def get_most_helpful_reviewer(df):
    # Group by the reviewerName and upvotes sum to create a Series
    grouped = df.groupby(by='reviewerID')['upvotes'].sum()

    # Get the entry that had the max number of upvotes then extract the user
    max_upvotes = max(grouped)
    # print max_upvotes
    user = grouped[grouped == max_upvotes].index[0]
    write_text_tofile("Most Helpful Reviewer: " + str(user) + ", Upvotes: " + str(max_upvotes))
    return user, max_upvotes


def merge_metadata(df):
    global merged_df
    # We use a different dataset to link the reviews and metadata. Original dataframe didn't have this info.
    # df1 = get_df('reviews_Amazon_Instant_Video.json.gz')
    df2 = get_df(METADATA_FILE, metadata=True)
    df3 = pd.merge(df, df2, on='asin', how='outer')
    merged_df = df3
    return df3


def get_most_and_least_expensive_high_review_product(df):
    """
    Retrieves the prices of the most expensive and least expensive
    products that got a rating >= 4
    :param df: Dataframe containing review and metadata info
    :return: dict containing {most_exp_product: price, least_exp_product: price}
    """
    df3 = merge_metadata(df)
    product_filter = df3['overall'] >= 4.0
    high_reviewed_products = df3[product_filter]
    # print high_reviewed_products[:10]
    # The data contained NaN so we use the nanmax/min funtions to get max/min
    most_exp = round(np.nanmax(high_reviewed_products['price'])[0], 2)
    least_exp = round(np.nanmin(high_reviewed_products['price'])[0], 2)

    most_exp_prod = df3.loc[df3['price'] == most_exp, 'asin'].iloc[0]
    least_exp_prod = df3.loc[df3['price'] == least_exp, 'asin'].iloc[0]
    write_text_tofile("Most Expensive Product: " + str(most_exp_prod) + ", Price: " + str(most_exp))
    write_text_tofile("Least Expensive Product: " + str(least_exp_prod) + ", Price: " + str(least_exp))
    return {most_exp_prod: most_exp, least_exp_prod: least_exp}


def cluster_product_data(df):
    # filter out NaNs

    # Create a dataframe to infer product information
    product_df = df[['asin', 'total_upvotes', 'average_rating']].copy()
    product_df = product_df.drop_duplicates('asin').set_index('asin')

    # Remove outliers greater than 3 standard deviations
    product_df = product_df[product_df.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    # print product_df[:10]

    cluster_num = 4
    df = product_df.dropna()
    # df.reset_index(drop=True, inplace=True)

    df_without_asin = df.ix[:, ['total_upvotes', 'average_rating']]  # Extract the columns with numbers only
    df_without_asin = scale(df_without_asin)
    df_pca = PCA(n_components=2)

    existing_2d = df_pca.fit_transform(df_without_asin)

    existing_df_2d = pd.DataFrame(existing_2d)
    existing_df_2d.columns = ['PC1', 'PC2']
    # print existing_2d

    kmeans = MiniBatchKMeans(n_clusters=cluster_num)
    kmeans.fit(existing_2d)
    #
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # print labels
    colors = ["g.", "r.", "c.", "y.", "b."]

    for i in range(len(existing_2d)):
        # print("Datapoint:", existing_2d[i], "label:", labels[i])
        plt.plot(existing_2d[i][0], existing_2d[i][1], colors[labels[i]], markersize=10)
        if i == 10000:
            break

    plt.xlabel("PC1")
    plt.ylabel("APC2")
    ax = plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    fig = ax.get_figure()
    fig.savefig(os.path.join(script_dir, 'cluster_product_data.jpg'))
    fig.clf()
    # plt.show()


def cluster_user_data(df):
    cluster_num = 5

    df = df.dropna()
    # df_wo_reviewerID = df.ix[:, ['total_upvotes', 'user_avg_rating']]  # Extract the columns with numbers only
    df_wo_reviewerID = df[["total_upvotes", "user_avg_rating"]].copy()
    df_wo_reviewerID = scale(df_wo_reviewerID)
    # print df_wo_reviewerID
    # df_pca = PCA(n_components=2)

    # existing_2d = df_pca.fit_transform(df_wo_reviewerID)
    #
    # existing_df_2d = pd.DataFrame(existing_2d)
    # existing_df_2d.columns = ['PC1', 'PC2']

    user_kmeans = MiniBatchKMeans(n_clusters=cluster_num, max_iter=100)
    user_kmeans.fit(df_wo_reviewerID)
    #
    centroids = user_kmeans.cluster_centers_
    labels = user_kmeans.labels_

    # print labels
    colors = ["g.", "r.", "c.", "y.", "b."]

    for i in range(len(df_wo_reviewerID)):
        # print("Datapoint:", existing_2d[i], "label:", labels[i])
        plt.plot(df_wo_reviewerID[i][0], df_wo_reviewerID[i][1], colors[labels[i]], markersize=10)
        if i == 10000:
            break

    plt.xlabel("Total Upvotes (Scaled)")
    plt.ylabel("Average User Rating (Scaled)")
    ax = plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    fig = ax.get_figure()
    fig.savefig(os.path.join(script_dir, 'cluster_user_data.jpg'))
    fig.clf()
    # plt.show()


def cluster_product_without_pca(df):
    # Create a dataframe to infer product information
    product_df = df[['asin', 'total_upvotes', 'average_rating']].copy()
    product_df = product_df.drop_duplicates('asin').set_index('asin')

    # Remove outliers greater than 3 standard deviations
    product_df = product_df[product_df.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    # print product_df[:10]

    cluster_num = 5
    df = product_df.dropna()
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    df_without_asin = df.ix[:, ['total_upvotes', 'average_rating']]  # Extract the columns with numbers only
    df_without_asin = scale(df_without_asin)  # This is not a dataframe, it's an nparray
    # df_pca = PCA(n_components=2)

    # existing_2d = df_pca.fit_transform(df_without_asin)

    # existing_df_2d = pd.DataFrame(existing_2d)
    # existing_df_2d.columns = ['PC1', 'PC2']
    # print existing_2d
    # print df_without_asin
    kmeans = MiniBatchKMeans(n_clusters=cluster_num)
    kmeans.fit(df_without_asin)
    #
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # print labels
    colors = ["g.", "r.", "c.", "y.", "b."]

    figure = plt.figure()
    for i in range(len(df_without_asin)):
        # print("Datapoint:", existing_2d[i], "label:", labels[i])
        plt.plot(df_without_asin[i][0], df_without_asin[i][1], colors[labels[i]], markersize=10)
        if i == 10000:
            break

    plt.xlabel("Total Upvotes (Scaled)")
    plt.ylabel("Average Rating (Scaled)")
    ax = plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    fig = ax.get_figure()
    fig.savefig(os.path.join(script_dir, 'cluster_product_data_without_pca.jpg'))
    fig.clf()
    # plt.show()


def get_elbow_plot(df, cols_to_keep, name='elbow_plot.jpg'):
    df_mod = df.ix[:, cols_to_keep]
    df_mod = scale(df_mod)
    K = range(1, 10)
    KM = [MiniBatchKMeans(n_clusters=k).fit(df_mod) for k in K]
    centroids = [k.cluster_centers_ for k in KM]

    D_k = [cdist(df_mod, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D, axis=1) for D in D_k]
    dist = [np.min(D, axis=1) for D in D_k]
    avgWithinSS = [sum(d) / df_mod.shape[0] for d in dist]

    ##### plot ###
    kIdx = 4

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    fig = ax.get_figure()
    fig.savefig(os.path.join(script_dir, name))
    fig.clf()
    # plt.show()


def compare_score_and_helpfulness(df):
    compare_df = df.groupby(['overall'])['upvotes'].sum()
    # print compare_df
    # print list(compare_df.columns.values)
    # ax = compare_df[['']]
    fig = plt.figure()

    ax = compare_df.plot(kind='bar', title='Number of Upvotes for each review score', fontsize=12)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Number of Upvotes')
    fig = ax.get_figure()
    fig.savefig(os.path.join(script_dir, 'compare_score_and_helpfulness.jpg'))
    fig.clf()
    # plt.show()

    # fig.savefig('compare.jpg')


def explo_reviewlen_helpfulness(df):
    df = df[(df.upvotes > 50)]
    # print df
    mod_df = df.groupby(['helpfulness'])['review_length'].mean()
    # print mod_df
    ax = mod_df.plot(kind='line', title='Avg Review Length vs Helpfulness', fontsize=12) #, ylim=(-0.2, 1.2))
    ax.set_xlabel('Helpfulness')
    ax.set_ylabel('Average Review Length')
    fig = ax.get_figure()
    fig.savefig(os.path.join(script_dir, 'explo_reviewlen_helpfulness.jpg'))
    fig.clf()
    # plt.show()


def explo_summarylen_helpfulness(df):
    df = df[(df.upvotes > 50)]
    mod_df = df.groupby(['helpfulness'])['summary_length'].mean()
    # print mod_df
    ax = mod_df.plot(kind='line', title='Avg Summary Length vs Helpfulness', fontsize=12)
    ax.set_xlabel('Helpfulness')
    ax.set_ylabel('Average Summary Length')
    fig = ax.get_figure()
    fig.savefig(os.path.join(script_dir, 'explo_summarylen_helpfulness.jpg'))
    fig.clf()
    # plt.show()


def explo_price_numreviews(df):
    global merged_df
    if merged_df is None:
        merge_metadata(df)

    local_merge = merged_df.dropna()
    local_merge = local_merge[["asin", "price"]].copy()
    local_merge.price = local_merge.price.round()
    local_merge = local_merge["price"].value_counts().sort_index()
    # print type(local_merge)
    # local_merge = local_merge.groupby(["price"])['asin'].sum()
    # print local_merge[:10]
    ax = local_merge.plot(kind='bar', title='Number of Reviews vs Price of Items', fontsize=12, figsize=(10, 7))
    ax.set_xlabel('Item Price')
    ax.set_ylabel('Number of Reviews')
    ax.set_xticks(ax.get_xticks()[::2])
    fig = ax.get_figure()
    fig.savefig(os.path.join(script_dir, 'explo_price_numreviews.jpg'))
    fig.clf()
    # plt.show()


def explo_reviewer_analysis(df):
    # Filter out reviewers who had an average rating < 3 but still rated something over 3
    df = df[(df.user_avg_rating < 3) & (df.overall > 3)]
    mod_df = df[["user_avg_rating", "overall"]].copy()
    mod_df = mod_df["user_avg_rating"].value_counts().sort_index()
    # print mod_df
    ax = mod_df.plot(kind='line', title='# of Reviews Rated > 3 by critical reviewers', fontsize=12)
    ax.set_xlabel('User Avg. Rating')
    ax.set_ylabel('# of Reviews Rated > 3')
    fig = ax.get_figure()
    fig.savefig(os.path.join(script_dir, 'explo_reviewer_analysis.jpg'))
    fig.clf()
    # plt.show()


if __name__ == '__main__':
    # print "Getting Product Means..."
    # get_product_means(df)
    # print "Getting Product Mode..."
    # get_mode_reviewed(df)
    # print "Getting Product Median..."
    # get_median_reviewed(df)
    # print "Getting Review Distribution..."
    # get_review_distribution(df)
    #
    # print "Getting most and least expensive products..."
    # get_most_and_least_expensive_high_review_product(df)
    # new_df = merged_df[['upvotes', 'overall']].copy()

    # Add in some extra columns to the dataset based on the analysis we want to perform
    df['total_upvotes'] = df['upvotes'].groupby(df['asin']).transform('sum')
    df['average_rating'] = df['overall'].groupby(df['asin']).transform('mean')
    df['user_avg_rating'] = df['overall'].groupby(df['reviewerID']).transform('mean')
    df['avg_helpfulness'] = df['helpfulness'].groupby(df['asin']).transform('mean')

    print df.info(memory_usage='deep')
    sys.exit()

    print "Clustering Product Data..."
    cluster_product_data(df)
    cluster_product_without_pca(df)
    get_elbow_plot(df, ['total_upvotes', 'average_rating'])

    print "Clustering Reviewer Data..."
    # Create dataframe to infer reviewer information
    reviewer_df = df[['reviewerID', 'total_upvotes', 'user_avg_rating']].copy()
    reviewer_df = reviewer_df.drop_duplicates('reviewerID').set_index('reviewerID')
    reviewer_df = reviewer_df[reviewer_df.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    # print reviewer_df[:10]
    cluster_user_data(reviewer_df)
    get_elbow_plot(reviewer_df, ['total_upvotes', 'user_avg_rating'], name='reviewer_elbow_plot.jpg')

    # Comparing the score and helpfulness attributes to find a correlation
    print "Comparing Score and Helpfulness..."
    compare_score_and_helpfulness(df)

    print "Performing Exploratory Analysis"
    # Perform some exploratory analysis
    explo_reviewlen_helpfulness(df)
    explo_summarylen_helpfulness(df)
    explo_price_numreviews(df)
    explo_reviewer_analysis(df)





