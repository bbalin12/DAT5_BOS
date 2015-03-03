# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:22:17 2015

@author: harishkashyap
"""

import pandas as pd
import numpy as np
import pylab as pl

df = pd.read_csv("/Users/harishkashyap/Documents/beer_reviews/beer_reviews.csv")
df.head()

beer_1, beer_2 = "Dale's Pale Ale", "Fat Tire Amber Ale"

beer_1_reviewers = df[df.beer_name==beer_1].review_profilename.unique()
beer_2_reviewers = df[df.beer_name==beer_2].review_profilename.unique()
common_reviewers = set(beer_1_reviewers).intersection(beer_2_reviewers)
print "Users in the sameset: %d" % len(common_reviewers)
list(common_reviewers)[:10]

def get_beer_reviews(beer, common_users):
    mask = (df.review_profilename.isin(common_users)) & (df.beer_name==beer)
    reviews = df[mask].sort('review_profilename')
    reviews = reviews[reviews.review_profilename.duplicated()==False]
    return reviews
beer_1_reviews = get_beer_reviews(beer_1, common_reviewers)
beer_2_reviews = get_beer_reviews(beer_2, common_reviewers)

cols = ['beer_name', 'review_profilename', 'review_overall', 'review_aroma', 'review_palate', 'review_taste']
beer_2_reviews[cols].head()

# choose your own way to calculate distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from scipy.stats.stats import pearsonr


ALL_FEATURES = ['review_overall', 'review_aroma', 'review_palate', 'review_taste']
def calculate_similarity(beer1, beer2):
    # find common reviewers
    beer_1_reviewers = df[df.beer_name==beer1].review_profilename.unique()
    beer_2_reviewers = df[df.beer_name==beer2].review_profilename.unique()
    common_reviewers = set(beer_1_reviewers).intersection(beer_2_reviewers)

    # get reviews
    beer_1_reviews = get_beer_reviews(beer1, common_reviewers)
    beer_2_reviews = get_beer_reviews(beer2, common_reviewers)
    dists = []
    for f in ALL_FEATURES:
        dists.append(euclidean_distances(beer_1_reviews[f], beer_2_reviews[f])[0][0])
    
    return dists

calculate_similarity(beer_1, beer_2)

# calculate only a subset for the demo
beers = ["Dale's Pale Ale", "Sierra Nevada Pale Ale", "Michelob Ultra",
         "Natural Light", "Bud Light", "Fat Tire Amber Ale", "Coors Light",
         "Blue Moon Belgian White", "60 Minute IPA", "Guinness Draught"]

# calculate everything for real production
# beers = df.beer_name.unique()

simple_distances = []
for beer1 in beers:
    print "starting", beer1
    for beer2 in beers:
        if beer1 != beer2:
            row = [beer1, beer2] + calculate_similarity(beer1, beer2)
            simple_distances.append(row)
            
cols = ["beer1", "beer2", "overall_dist", "aroma_dist", "palate_dist", "taste_dist"]
simple_distances = pd.DataFrame(simple_distances, columns=cols)
simple_distances.tail()

def calc_distance(dists, beer1, beer2, weights):
    mask = (dists.beer1==beer1) & (dists.beer2==beer2)
    row = dists[mask]
    row = row[['overall_dist', 'aroma_dist', 'palate_dist', 'taste_dist']]
    dist = weights * row
    return dist.sum(axis=1).tolist()[0]

weights = [2, 1, 1, 1]
print calc_distance(simple_distances, "Fat Tire Amber Ale", "Dale's Pale Ale", weights)
print calc_distance(simple_distances, "Fat Tire Amber Ale", "Michelob Ultra", weights)

my_beer = "Coors Light"
results = []
for b in beers:
    if my_beer!=b:
        results.append((my_beer, b, calc_distance(simple_distances, my_beer, b, weights)))
sorted(results, key=lambda x: x[2])

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

