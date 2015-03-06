# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:22:17 2015

@author: harishkashyap
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pprint as pp
import os

filename = os.path.join("/Users/harishkashyap/Documents/beer_reviews/beer_reviews.csv")
df = pd.read_csv(filename)
# let's limit things to the top 250
n = 250
top_n = df.beer_name.value_counts().index[:n]
df = df[df.beer_name.isin(top_n)]

print "melting..."
df_wide = pd.pivot_table(df, values=["review_overall"],
                         rows=["beer_name", "review_profilename"],
                         aggfunc=np.mean).unstack()

# any cells that are missing data (i.e. a user didn't buy a particular product)
# we're going to set to 0
df_wide = df_wide.fillna(0)

# this is the key. we're going to use cosine_similarity from scikit-learn
# to compute the distance between all beers
print "calculating similarity"
dists = cosine_similarity(df_wide)

# stuff the distance matrix into a dataframe so it's easier to operate on
dists = pd.DataFrame(dists, columns=df_wide.index)

# give the indicies (equivalent to rownames in R) the name of the product id
dists.index = dists.columns

def get_sims(products):
    """
    get_top10 takes a distance matrix an a productid (assumed to be integer)
    and will calculate the 10 most similar products to product based on the
    distance matrix

    dists - a distance matrix
    product - a product id (integer)
    """
    p = dists[products].apply(lambda row: np.sum(row), axis=1)
    p = p.order(ascending=False)
    return p.index[p.index.isin(products)==False]

get_sims(["Sierra Nevada Pale Ale", "120 Minute IPA", "Coors Light"])
# Index([u'Samuel Adams Boston Lager', u'Sierra Nevada Celebration Ale', u'90 Minute IPA', u'Arrogant Bastard Ale', u'Stone IPA (India Pale Ale)', u'60 Minute IPA', u'HopDevil Ale', u'Stone Ruination IPA', u'Sierra Nevada Bigfoot Barleywine Style Ale', u'Storm King Stout', u'Samuel Adams Winter Lager', u'Samuel Adams Summer Ale', u'Prima Pils', u'Anchor Steam Beer', u'Old Rasputin Russian Imperial Stout', u'Samuel Adams Octoberfest', ...], dtype='object')

from yhat import Yhat, YhatModel, preprocess

class BeerRecommender(YhatModel):
    @preprocess(in_type=dict, out_type=dict)
    def execute(self, data):
        beers = data.get("beers")
        suggested_beers = get_sims(beers)
        result = []
        for beer in suggested_beers:
            result.append({"beer": beer})
        return result

BeerRecommender().execute({"beers": ["Sierra Nevada Pale Ale","120 Minute IPA", "Stone Ruination IPA"]})

yh = Yhat("USERNAME", "APIKEY", "http://cloud.yhathq.com")
yh.deploy("BeerRecommender", BeerRecommender, globals())

yh.predict("BeerRecommender",{"beers": ["Sierra Nevada Pale Ale","120 Minute IPA", "Stone Ruination IPA"]})