# Ansel Bobrow and Kirin Mueller
# Language and Computation I Final Project

from __future__ import division

#############################################################################
# Copyright 2011 Jason Baldridge
# 
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#############################################################################

# Imports from Python standard libraries
import re,math
import pandas
from operator import itemgetter

# Imports from external packages included in the same directory
from porter_stemmer import PorterStemmer
import twokenize

# Imports from other packages created for this homework
from classify_util import makefeat,window

#############################################################################
# Code to set up some resources for your features

# Create the stemmer for later use.
stemmer = PorterStemmer()

# Read in the stop words
stop_words = set([x.strip() for x in open("stopwords.english", encoding="ISO-8859-1").readlines()])

#############################################################################
# Add your regular expressions here.
# Added two regular expressions for use in the features.


#############################################################################
# General features (for both subjectivity and polarity)
def extract_features (song, extended_features):
    features = []

    # FIXME: Change to use tokenization from twokenize. FIXED!
    tokens = twokenize.tokenize(song.lyrics)

    # FIXME: lower case the tokens. FIXED!
    for token in tokens:
        token = token.lower()

    pandas.from_pickle("song-genre-classification/hiphop_df")

    # FIXME: Create stems here. FIXED!
    for token in tokens:
        token = (stemmer.stem_token(token))

    # Add unigram features. 
    # FIXME: exclude stop words. FIXED!
    for token in tokens:
        for stop_word in stop_words:
            if token == stop_word:
                tokens.remove(token)
    
    # FIXME: consider using lower case version and/or stems. FIXED!
    features.extend([makefeat("word",tok) for tok in tokens])

    # The same thing, using a for-loop (boooring!)
    #for tok in tokens:
    #    features.append(makefeat("word",tok))
        
    if extended_features:
        # FIXME: Add bigram features (suggestion: use the window function in classify_utils.py)
        bigrams = window(tokens, 2)
        trigrams = window(tokens, 3)

        for bigram in bigrams:
            features.append(f"bigram="+str(bigram))

        for trigram in trigrams:
            features.append(f"trigram="+str(trigram))

        # FIXME: Add other features -- be creative!
        apostrophe_count = 0
        comma_count = 0
        period_count = 0
        exclamation_count = 0
        question_count = 0
        semicolon_count = 0
        colon_count = 0
        hyphen_count = 0
        Xx_count = 0
        XX_count = 0

        for token in tokens:
            if token == "'":
                apostrophe_count += 1
            if token == ",":
                comma_count += 1
            if token == ".":
                period_count += 1
            if token == "!":
                exclamation_count += 1
            if token == "?":
                question_count += 1
            if token == ":":
                colon_count += 1
            if token == ";":
                semicolon_count += 1
            if token == "-":
                hyphen_count += 1

            if token[0].isupper():
                if token.isupper():
                    XX_count += 1
                else:
                    Xx_count += 1

        # add features for positive/negative words, positive and negative numbers, numbers of capitalized words
        features.append("apostrophe="+str(apostrophe_count))
        features.append("comma="+str(comma_count))
        features.append("period="+str(period_count))
        features.append("exclamation="+str(exclamation_count))
        features.append("question="+str(question_count))
        features.append("colon="+str(colon_count))
        features.append("semicolon="+str(semicolon_count))
        features.append("hyphen="+str(hyphen_count))

        features.append("Xx_count="+str(Xx_count))
        features.append("XX_count="+str(XX_count))

    return features


#############################################################################
# Predict sentiment based on ratio of positive and negative terms in a tweet
def majority_class_baseline (tweetset):

    # FIXME: Compute the most frequent label in tweetset and return it. FIXED!

    # Create dictionary of labels, and add up all the counts for each label
    labels = dict([("positive", 0), ("negative", 0), ("neutral", 0)])
    for tweet in tweetset:
        labels[tweet.label] += 1

    keys = list(labels.keys())
    frequencies = list(labels.values())

    # Find the maximum count among the labels, and return the label for that maximum count
    majority_class_label = keys[frequencies.index(max(labels["positive"], labels["negative"], labels["neutral"]))]

    return majority_class_label


#############################################################################
# Predict sentiment based on ratio of positive and negative terms in a tweet
def lexicon_ratio_baseline (tweet):

    # FIXME: Change to use tokenization from twokenize. FIXED!
    tokens = twokenize.tokenize(tweet.content)

    # FIXME: Count the number of positive and negative words in the tweet FIXED!
    num_positive = 0
    num_negative = 0

    for word in tokens: 
        for pos_word in pos_words:
            if word == pos_word:
                num_positive += 1
        for neg_word in neg_words:
            if word == neg_word:
                num_negative += 1

    #########################################################################
    # Don't change anything below this comment
    #########################################################################

    # Let neutral be prefered if nothing is found.
    num_neutral = .2

    # Go with neutral if pos and neg are the same
    if num_positive == num_negative:
        num_neutral += len(tokens)

    # Add a small count to each so we don't get divide-by-zero error
    num_positive += .1
    num_negative += .1

    denominator = num_positive + num_negative + num_neutral

    # Create pseudo-probabilities based on the counts
    predictions = [("positive", num_positive/denominator), 
                   ("negative", num_negative/denominator),
                   ("neutral", num_neutral/denominator)]

    # Sort
    predictions.sort(key=itemgetter(1),reverse=True)

    # Return the top label and its confidence
    return predictions[0]
    
