#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/7 14:48
# @Author  : FywOo02
# @FileName: model.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sklearn.preprocessing
import sklearn.cluster
pd.set_option('display.max_columns', None)

# load data
data = pd.read_csv('air_data.csv')
# print(data.shape)
# print(data.head())
# print(data.describe().T)
data = data.drop_duplicates()  # remove duplicate value
# print(data.isnull().sum())


# feature engineering
# use LRFMC model to do feature screening
''' LRFMC
Length of Relationship: The length of the customer relationship, reflecting the possible length of activity.
Recency: The time interval between recent purchases, reflecting the current active status.
Frequency: The frequency of customer spending, reflecting the customer's loyalty.
Mileage: The total number of miles flown by the customer, reflecting the customer's dependence on the flight.
Coefficient of Discount: The average discount rate enjoyed by the customer, which reflects the value of the customer.
'''
load_time = datetime.datetime.strptime('2014/03/31', '%Y/%m/%d')

ffp_dates = []
for ffp_date in data['FFP_DATE']:
    ffp_dates.append(datetime.datetime.strptime(ffp_date, '%Y/%m/%d'))

length_of_relationship = []
for ffp_date in ffp_dates:
    length_of_relationship.append((load_time - ffp_date).days)

data['LEN_REL'] = length_of_relationship

features = ['LEN_REL', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']
data = data[features]
features = ['L', 'R', 'F', 'M', 'C']
data.columns = features

# print(data.head())
# print(data.describe().T)

# Feature Normalization
data = (data - data.mean(axis=0)) / (data.std(axis=0))
ss = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)  # Normalization
data = ss.fit_transform(np.array(data))  # Data Convert
data = pd.DataFrame(data, columns=features)

# print(data.describe().T)

# Clustering
# Algorithm: K-means
from sklearn.cluster import KMeans

K = KMeans(n_clusters=5)
K.fit(data)


def get_cluster(data):
    labels = pd.Series(K.labels_)
    nums = labels.value_counts().sort_index()
    types = pd.Series(['Costumer Cluster' + str(i) for i in range(1, 6)])
    centers = pd.DataFrame(K.cluster_centers_, columns=data.columns)
    new_data = pd.concat([types, nums, centers], axis='columns')
    new_data.columns = ['cluster_names', 'cluster_num', 'ZL', 'ZR', 'ZF', 'ZM', 'ZC']

    return new_data


data = get_cluster(data)
# print(data)


# Customer Value Visualization
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_radar(data):
    colors = ['tomato', 'darkorange', 'limegreen', 'darkcyan', 'royalblue']
    names = data['cluster_names'].tolist()
    labels = data.columns.tolist()[2:]
    centers = pd.concat([data.iloc[:, 2:], data.iloc[:, 2]], axis=1)
    centers = np.array(centers)
    n = len(labels)
    labels = data.iloc[:, 1:].columns
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, polar=True)

    floor = np.floor(centers.min())
    ceil = np.ceil(centers.max())

    for i in range(n):
        ax.plot([angles[i], angles[i]], [floor, ceil], lw=0.5, color='grey')

    for i in range(len(names)):
        ax.plot(angles, centers[i], colors[i], label=names[i])
        plt.fill(angles, centers[i], facecolor=colors[i], alpha=0.2)

    ax.set_thetagrids(angles * 180 / np.pi, labels)
    plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0.0))

    ax.set_theta_zero_location('N')
    ax.spines['polar'].set_visible(False)

    plt.show()


plot_radar(data)
