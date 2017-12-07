# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# subway=pd.read_csv('aa.csv')

city = pd.read_csv('/home/qiwei/workspace/python/TianChi/weather/CityData.csv')

trainData = pd.read_csv(
    '/home/qiwei/workspace/python/TianChi/weather/In_situMeasurementforTraining_20171205/In_situMeasurementforTraining_201712.csv'
)


def get_hour_data(hour):
    showData = trainData[(trainData.date_id == 1) & (trainData.hour == hour)]
    dpoint = showData[showData.wind > 14]
    return dpoint


def get_hour_data_3d():
    #showData=trainData[trainData.date_id==1dd]
    showData = trainData[(trainData.date_id == 1) & (trainData.hour < 10)]
    dpoint = showData[showData.wind > 14]
    return dpoint


def draw_map(dpoint, hour):
    # plt.plot(subway['xid'],subway['yid'],'go')
    plt.plot(city['xid'], city['yid'], 'rx')
    plt.plot([142], [328], 'bo')
    # plt.plot(city.ix[[0,1],'xid'],city.ix[[0,1],'yid'],ls='--')
    #plt.plot(subway['xid'],subway['yid'],'go')
    plt.scatter(
        dpoint['xid'],
        dpoint['yid'],
        label='skitscat',
        color='k',
        s=25,
        marker="o")
    plt.title(hour)  # give plot a title
    plt.grid(True)
    plt.show()


def draw_map_3d(dpoint):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(dpoint['xid'], dpoint['yid'], dpoint['hour'], c='k', marker='o')

    plt.show()


if __name__ == '__main__':
    # dp=get_hour_data_3d()
    # draw_map_3d(dp)
    for i in range(3, 20):
        dp = get_hour_data(i)
        draw_map(dp, i)
