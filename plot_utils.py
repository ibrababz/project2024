# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:03:20 2022

@author: i_bab
"""

import matplotlib.pyplot as plt
def plot_test_train(train_loss_tracker, test_loss_tracker):
    
    fig, ax = plt.subplots(1)
    xdata  = range(len(train_loss_tracker))
    train_loss_plot =  train_loss_tracker
    ax.plot(xdata, train_loss_plot, label = 'train loss')
    test_loss_plot = test_loss_tracker
    ax.plot(xdata, test_loss_plot, label = 'test loss')
    ax.set_ylim()
    plt.title('Loss per epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show(fig)
    
    return fig