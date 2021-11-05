# -*- coding: utf-8 -*-
"""
Created on Tue 26 March 18:44:45 2020

@author: Mnemosyne

Vocal learning model results (plots of)
"""

import os
import time
import glob
import pickle
import numpy as np
import matplotlib
import librosa
from matplotlib import rcParams, cm, colors
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import scipy.io.wavfile as wav
csfont = {'fontname':'Times New Roman'}

from songbird_data_analysis import Song_functions

def magnitude(v):
    """
    :param v = (x,y,z): 3D cartesian coordinates - vector
    :return m: magnitude (Euclidian norm in this case)
    """
    m = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    return m

def polar_coord(v):
    """
    :param v = (x,y,z): 3D cartesian coordinates - vector
    :return r,phi, theta: polar coordinates
    """
    r = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    phi = np.arctan(v[1]/v[0])
    theta = np.arctan(np.sqrt(v[0]**2 + v[1]**2)/v[2])

    return r, phi, theta

def arctan_coord(v):
    """
    :param v: 3D cartesian coordinates - vector
    :return x_new, y_new: 2D vector with x_new = arctan(v0/v2) ane y_new = arctan(v0/v2)
    """

    x_new = np.arctan(v[0]/v[1])
    y_new = np.arctan(v[0]/v[2])

    return x_new, y_new

def arctan_distance(v,w):
    """
    :param v, w: vectors of the same size
    :return: "angular" distance component by componet - vector
    """
    d = np.zeros((np.size(v),))
    for i in range(0, np.size(v)):
        d[i] = np.arctan(v[i] - w[i])

    return d

def create_sphere(cx,cy,cz, r, resolution=360):
    '''
    create sphere with center (cx, cy, cz) and radius r
    '''
    phi = np.linspace(0, 2*np.pi, 2*resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r*np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return np.stack([x,y,z])

def plot_auditory_activation(args):
    """
    Plot the results of the different auditory activation functions (results from the test function)
    """
    # Repertoire
    classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']
    for sim_counter in range(0, args.N_sim):
        for cl in range(0, len(args.classifier_name)):
            print(args.classifier_name[cl])
            softmax_sum_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_softmax_sum_expl_' + str(sim_counter) + '.npy')
            softmax_mean_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_softmax_mean_expl_' + str(sim_counter) + '.npy')
            raw_score_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_raw_score_expl_' + str(sim_counter) + '.npy')
            raw_mean_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_mean_expl_' + str(sim_counter) + '.npy')
            mean_norm_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_mean_norm_expl_' + str(sim_counter) + '.npy')
            logistic_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_logistic_expl_' + str(sim_counter) + '.npy')
            tanh_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_tanh_expl_' + str(sim_counter) + '.npy')
            minmax_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_minmax_expl_' + str(sim_counter) + '.npy')
            sign_minmax_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_sign_minmax_expl_' + str(sim_counter) + '.npy')
            sign_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_sign_expl_' + str(sim_counter) + '.npy')
            square_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_square_expl_' + str(sim_counter) + '.npy')
            arctg_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_arctg_expl_' + str(sim_counter) + '.npy')
            scaling_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_scaling_expl' + str(sim_counter) + '.npy', allow_pickle=True)
            scaling_softmax_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_scaling_softmax_expl' + str(sim_counter) + '.npy', allow_pickle=True)
            softmax_MAX_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_softmax_MAX_expl' + str(sim_counter) + '.npy', allow_pickle=True)
            max_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_max_expl' + str(sim_counter) + '.npy', allow_pickle=True)
            max_norm_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_max_norm_expl' + str(sim_counter) + '.npy', allow_pickle=True)
            p95_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_p95_expl' + str(sim_counter) + '.npy', allow_pickle=True)

            for i in range(0, np.shape(raw_score_expl)[0]):
                for j in range(0, len(classes)):
                    if p95_expl[i,j] > 1:
                        p95_expl[i,j] = 1

            # Time vector
            x_time = np.linspace(0, np.shape(raw_score_expl)[0], np.shape(raw_score_expl)[0])

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(raw_score_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i, j].set_ylim(0, 1000)
                    ax[i, j].set_xlabel('MinMax score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_raw_score_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(p95_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h, width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(-0.1, 1)
                    ax[i, j].set_ylim(0, 1500)
                    ax[i, j].set_xlabel('p95 score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_p95_expl_pw' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(max_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h, width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i, j].set_ylim(0, 1000)
                    ax[i, j].set_xlabel('Max score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_max_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(max_norm_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h, width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i, j].set_ylim(0, 1000)
                    ax[i, j].set_xlabel('Max norm score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_max_norm_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(scaling_softmax_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h, width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i, j].set_ylim(0, 1000)
                    ax[i, j].set_xlabel('Scaling softmax score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_scaling_softmax_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(softmax_MAX_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h, width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i, j].set_ylim(0, 1000)
                    ax[i, j].set_xlabel('Softmax MAX score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_softmax_MAX_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(scaling_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h, width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(-0.1, 1)
                    ax[i, j].set_ylim(0, 1500)
                    ax[i, j].set_xlabel('Scaling score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_scaling_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(arctg_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(-1, 1)
                    ax[i, j].set_xlabel('Arctg score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_arctg_expl' + str(
                    sim_counter) + '.' + args.format)


            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(square_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(-1, 1)
                    ax[i, j].set_xlabel('Square root score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_square_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(sign_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(-1, 1)
                    ax[i, j].set_xlabel('Sign score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_sign_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(minmax_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(-1, 1)
                    ax[i, j].set_xlabel('Minmax score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_minmax_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(sign_minmax_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i,j].set_ylim(0,800)
                    ax[i, j].set_xlabel('Sign minmax score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_sign_minmax_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(logistic_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h, width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i, j].set_xlabel('Logistic score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_logistic_expl_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(tanh_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(-1, 1)
                    ax[i, j].set_xlabel('Tanh score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_tanh_expl_expl' + str(
                    sim_counter) + '.' + args.format)


            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(raw_mean_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i, j].set_xlabel('Raw mean score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_raw_mean_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(mean_norm_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i, j].set_xlabel('Mean score', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_mean_norm_expl' + str(
                    sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    h, bins = np.histogram(softmax_sum_expl[:, 4 * i + j], bins=15)
                    ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
                    ax[i, j].set_xlim(0, 1)
                    ax[i, j].set_xlabel('Soft-max', fontsize=8)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_softmax_sum_expl' + str(
                    sim_counter) + '.' + args.format)

            plt.close('all')

            for b in range(0, np.size(args.beta)):
                fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                for i in range(0, 4):
                    for j in range(0, 4):
                        h, bins = np.histogram(softmax_mean_expl[b][:, 4 * i + j], bins=15)
                        ax[i, j].bar(bins[:-1], h , width=0.05, color='b', alpha=0.6)
                        ax[i, j].spines['top'].set_color('none')
                        ax[i, j].spines['right'].set_color('none')
                        ax[i, j].set_xlim(0, 1)
                        ax[i, j].set_xlabel('Raw score', fontsize=8)
                        ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                plt.tight_layout()
                plt.savefig(
                    args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[
                        cl] + '_softmax_mean_expl_beta_' + str(args.beta[b]) + '_' + str(
                        sim_counter) + '.' + args.format)

    print('Done')

def plot_sensory(args):
    """
    Plots of the results obtained from the leanring model (VLM function).
    """
    # Colors
    color = ['r', 'b', 'k', 'orange', 'magenta', 'purple']

    # Repertoire
    classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

    p95_mean = np.zeros((len(args.learning_rate), args.n_points + 1, len(classes)))
    for lr in range(0, len(args.learning_rate)):
        print(args.learning_rate[lr])
        for cl in range(0, len(args.classifier_name)):
            print(args.classifier_name[cl])
            p95_all_sim = []
            for sim_counter in range(0, args.N_sim):
                p95 = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_p95_sim_' + str(sim_counter) + '.npy')
                p95_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_p95_expl_' + str(sim_counter) + '.npy')

                # Focus on 200 time steps
                p95_focus = p95[0:200, :]

                # Remove focus (every N points up to 200 points) - CHECK PLOT
                p95_begin = p95[0:200, :]
                p95_jump = np.zeros((args.n_points + 1, np.size(args.T_names)))
                p95_jump[0:14, :] = p95_begin[0::15, :]
                p95_jump[14::, :] = p95[200::, :]

                # All sim vector
                p95_all_sim.append(p95_jump)

                # Time vector
                x_time = np.linspace(0, args.MAX_trial, np.shape(p95_jump)[0])
                x_time_expl = np.linspace(0, np.shape(p95_expl)[0], np.shape(p95_expl)[0])
                x_time_focus = np.linspace(0, np.shape(p95_focus)[0], np.shape(p95_focus)[0])

                fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                for i in range(0, 4):
                    for j in range(0, 4):
                        ax[i, j].plot(x_time_focus, p95_focus[:, 4 * i + j], 'b')
                        ax[i, j].set_ylim(0, 1)
                        ax[i, j].set_xlim(0, np.shape(p95_focus)[0])
                        ax[i, j].set_ylabel('Average A', fontsize=8)
                        ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                        ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                        ax[i, j].spines['top'].set_color('none')
                        ax[i, j].spines['right'].set_color('none')
                plt.tight_layout()
                plt.savefig(
                    args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                        args.learning_rate[lr]) + '_p95_FOCUS_sim' + str(
                        sim_counter) + '.' + args.format)

                W_p95 = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_W_p95_sim_' + str(sim_counter) + '.npy')[0:args.MAX_trial, :, :]

                # Plot the evolution of the synaptic weights over trials
                if np.size(args.T_names) == len(classes):
                    fig, ax = plt.subplots(4, 4, sharex='col', sharey='row', figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            for k in range(0, args.wavegan_latent_dim):
                                ax[i, j].plot(x_time_expl, W_p95[:, k, 4 * i + j], color[k])
                            ax[i, j].set_ylabel('Weights', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                            ax[i,j].set_ylim(-1,1)
                    plt.tight_layout()
                    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + 'Synaptic_weights_evolution_p95' + str(sim_counter) + '.' + args.format)

                # Plot activation of the exploration
                fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                for i in range(0, 4):
                    for j in range(0, 4):
                        ax[i, j].plot(x_time_expl, p95_expl[:, 4 * i + j], 'b')
                        #ax[i, j].set_ylim(0, 1)
                        ax[i, j].set_ylabel('Average A', fontsize=8)
                        ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                        ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                plt.tight_layout()
                plt.savefig(
                    args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                        args.learning_rate[lr]) + '_p95_expl' + str(
                        sim_counter) + '.' + args.format)

                # Plot activation during learning
                fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                for i in range(0, 4):
                    for j in range(0, 4):
                        ax[i, j].plot(x_time, p95_all_sim[sim_counter][:, 4 * i + j], 'b')
                        ax[i, j].set_ylim(0, 1)
                        ax[i, j].set_xlim(0, args.MAX_trial-1)
                        ax[i, j].set_ylabel('Average A', fontsize=8)
                        ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                        ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                        ax[i, j].spines['top'].set_color('none')
                        ax[i, j].spines['right'].set_color('none')
                plt.tight_layout()
                plt.savefig(
                    args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                        args.learning_rate[lr]) + '_p95_sim' + str(
                        sim_counter) + '.' + args.format)

                # [TODO] add comment here when I try this option
                if args.example == True:
                    if sim_counter == 1:
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharey=True, sharex=True)
                        for lr in range(0, len(args.learning_rate)):
                            ax.plot(x_time, p95_all_sim[sim_counter][:, 14], 'b')
                        ax.spines['top'].set_color('none')
                        ax.spines['right'].set_color('none')
                        ax.set_xlim(0, args.MAX_trial)
                        ax.set_xlabel('Time (in number of time steps)', fontsize=15)
                        ax.set_ylabel('Activation', fontsize=15)

                        plt.savefig(
                            args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                                args.learning_rate[lr]) + '_R' + '.' + args.format)

                plt.close('all')

            # Average over multiple simulations
            p95_mean_sim = np.mean(p95_all_sim, axis=0)
            p95_mean[lr, :, :] = p95_mean_sim

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for sim_counter in range(0, args.N_sim):
                for i in range(0, 4):
                    for j in range(0, 4):
                        #ax[i, j].plot(x_time, np.ones((np.shape(p95)[0], 1)), 'k')
                        ax[i, j].plot(x_time, p95_all_sim[sim_counter][:, 4 * i + j], c=color[sim_counter], alpha=.7)
                        ax[i, j].set_ylim(0, 1)
                        ax[i, j].set_xlim(0, args.MAX_trial)
                        ax[i, j].set_ylabel('Average A', fontsize=8)
                        ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                        ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                        ax[i, j].spines['top'].set_color('none')
                        ax[i, j].spines['right'].set_color('none')
                plt.tight_layout()
                plt.savefig(
                    args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                        args.learning_rate[lr]) + '_p95_sim_ALL' + '.' + args.format)

            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for sim_counter in range(0, args.N_sim):
                for i in range(0, 4):
                    for j in range(0, 4):
                        #ax[i, j].plot(x_time, np.ones((np.shape(p95)[0], 1)), 'k')
                        ax[i, j].plot(x_time, p95_mean_sim[:, 4 * i + j], c=color[sim_counter], alpha=.7)
                        ax[i, j].set_ylim(0, 1)
                        ax[i, j].set_xlim(0, args.MAX_trial)
                        ax[i, j].set_ylabel('Average A', fontsize=8)
                        ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                        ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                        ax[i, j].spines['top'].set_color('none')
                        ax[i, j].spines['right'].set_color('none')
                plt.tight_layout()
                plt.savefig(
                    args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                        args.learning_rate[lr]) + '_p95_MEAN' + '.' + args.format)

    # Comparison between different learning rates
    cfr_lr = ['10e-1', '10e-2']
    fig, ax = plt.subplots(4, 4, figsize=(12, 7))
    for lr in range(0, len(args.learning_rate)):
        for i in range(0, 4):
            for j in range(0, 4):
                ax[i, j].plot(x_time, p95_mean[lr,:, 4 * i + j], c=color[lr], alpha=.7, label=cfr_lr[lr])
                ax[i, j].set_ylim(0, 1)
                ax[i, j].set_xlim(0, args.MAX_trial)
                ax[0, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                ax[i, j].spines['top'].set_color('none')
                ax[i, j].spines['right'].set_color('none')
            ax[i, 0].set_ylabel('Average A', fontsize=8)
        ax[0, 0].legend(fontsize=5)
        plt.tight_layout()
        plt.savefig(
            args.data_dir + '/' + args.output_dir + '/' + '_p95_MEAN_all' + '.' + args.format)

    np.save(args.data_dir + '/' + 'p95_MEAN_lr_' + str(args.wavegan_latent_dim) + '.npy' ,p95_mean)

    plt.close('all')
    print('Done')

def cfr_dim13(p95_MEAN, colors, ld, args):
    """
    :param p95_MEAN: list of the arrays containing the data (one per latent space condition, two values each - one per learning rate condition)
    :return: figure with the comparison (one per leanring rate condition)
    """
    x_time = np.linspace(0, args.MAX_trial, 201)
    classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

    for lr in range(0, len(args.learning_rate)):
        fig, ax = plt.subplots(4, 4, figsize=(12, 7))
        for i in range(0, 4):
            for j in range(0, 4):
                for l in range(0, len(p95_MEAN)):
                    ax[i, j].plot(x_time, p95_MEAN[l][lr,:, 4 * i + j], c=colors[l], alpha=.7, label=str(ld[l]))
                ax[i, j].set_ylim(0, 1)
                ax[i, j].set_xlim(0, args.MAX_trial)
                ax[0, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                ax[i, j].spines['top'].set_color('none')
                ax[i, j].spines['right'].set_color('none')
            ax[i, 0].set_ylabel('Average A', fontsize=8)
        ax[0, 0].legend(fontsize=5)
        plt.tight_layout()
        plt.savefig(
            args.data_dir + '/' + '_p95_MEAN_lr_' + str(args.learning_rate[lr]) + '.' + args.format)

    plt.close('all')
    print('Done')

def plot_sensory_test(args):
    # Colors
    color = ['r', 'b', 'k', 'orange', 'magenta', 'purple']

    # Repertoire
    classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']

    for sim_counter in range(0, args.N_sim):
        cfr_class_A_all = []
        cfr_class_A_expl_all = []
        cfr_class_raw_all = []
        cfr_class_expl_all = []
        conv = []
        for cl in range(0, len(args.classifier_name)):
            print(args.classifier_name[cl])
            cfr_class_A = []
            cfr_class_A_expl = []
            cfr_class_raw = []
            cfr_class_expl = []
            mean_spectrogram_env = []
            T = []
            for lr in range(0, len(args.learning_rate)):
                print(args.learning_rate[lr])
                sensory_gen = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_A_sim_' + str(sim_counter) + '.npy')
                sensory_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_A_expl_' + str(sim_counter) + '.npy')
                sensory_expl_all = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_A_expl_all_' + str(sim_counter) + '.npy')
                raw_score = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) +  '_raw_score_sim_' + str(sim_counter) + '.npy')
                max_score = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) +  '_max_sim_' + str(sim_counter) + '.npy')
                max_norm = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) +  '_max_norm_sim_' + str(sim_counter) + '.npy')
                max_scaling = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) +  '_max_scaling_sim_' + str(sim_counter) + '.npy')
                raw_score_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_raw_score_expl_' + str(sim_counter) + '.npy')
                max_score_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_max_score_expl_' + str(sim_counter) + '.npy')
                max_norm_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_max_norm_expl_' + str(sim_counter) + '.npy')
                max_scaling_expl = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_max_scaling_expl_' + str(sim_counter) + '.npy')
                #cfr_class_A.append(sensory_gen)
                #cfr_class_A_expl.append(sensory_expl)
                cfr_class_raw.append(raw_score)
                cfr_class_expl.append(raw_score_expl)

                # Time vector
                x_time = np.linspace(0, args.MAX_trial, np.shape(raw_score)[0])
                x_time_expl = np.linspace(0, np.shape(raw_score_expl)[0], np.shape(raw_score_expl)[0])
                #

                # if args.learning_rate[lr] == 0.01:
                #     for c in range(0, np.size(args.T_names)):
                #         loc = np.where(raw_score[:, c] > 0.9)[0]
                #
                #         spectrograms_envelope = []
                #         for sp in range(0, np.size(loc)):
                #             samples_aux, sr = librosa.load(
                #                 args.data_dir + '/' + args.sim_name + str(sim_counter) + '/' + args.classifier_name[
                #                     cl] + '_lr' + str(args.learning_rate[lr]) + '_' + args.sim_name + str(
                #                     sim_counter) + '_' + str(
                #                     loc[sp]) + '/' + 'sensory_production_' + args.T_names[c] + '.wav', sr=16000)
                #             trim = librosa.effects.trim(samples_aux.astype(np.float), top_db=20)
                #             samples_aux = trim[0]
                #
                #             if samples_aux.size / 16 < 4000:
                #                 aux_size = 4000 - samples_aux.size / 16
                #                 silence = np.zeros((int(round(aux_size / 2) * 16)), )
                #                 samples_aux = np.append(silence, samples_aux)
                #                 samples_aux = np.append(samples_aux, silence)
                #
                #             rawsong = samples_aux.astype(float)
                #             rawsong = rawsong.flatten()
                #             amp = Song_functions.smooth_data(rawsong, sr, freq_cutoffs=(500, 7999))
                #
                #             # if args.T_names[c] == 'N':
                #             # new_song = rawsong[0:np.where(amp > 0.00001)[0][-1]]  # new training
                #             # silence = np.zeros((8000 - np.size(new_song),))
                #             # new_song = np.append(silence, new_song)
                #
                #             # else:
                #             new_song = rawsong[np.where(amp > 0.00001)[0][0]::]
                #             silence = np.zeros((100000 - np.size(new_song),))
                #             new_song = np.append(new_song, silence)
                #
                #             X = librosa.stft(new_song, n_fft=args.N, hop_length=args.H, win_length=args.N,
                #                              window='hann',
                #                              pad_mode='constant', center=True)
                #             T_coef = np.arange(X.shape[1]) * args.H / sr * 1000  # save to plot
                #             spectrograms_envelope.append(np.log(1 + 100 * np.abs(X ** 2)))
                #
                #         mean_spectrogram_env.append(np.mean(spectrograms_envelope, axis=0))  # dimension 16
                #         T.append(T_coef)
                #
                #     np.save(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                #         args.learning_rate[lr]) + 'Mean_spectrogram_envelope', mean_spectrogram_env)
                #
                #     # Mean spectrogram after convergence
                #     fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 14), sharey=True, sharex=True)
                #     for i in range(0, 4):
                #         for j in range(0, 4):
                #             extent = [0, np.max(T_coef[4 * i + j]), 0, 8000]
                #             if mean_spectrogram_env[4 * i + j].size > 1:
                #                 axs[i, j].imshow(mean_spectrogram_env[4 * i + j], extent=extent, cmap=args.color,
                #                                  aspect='auto', origin='lower',
                #                                  norm=colors.PowerNorm(gamma=0.5))  # gamma 0.2 in original data
                #             axs[i, j].set_title(args.T_names[4 * i + j], fontsize=15)
                #             # axs[i, j].set_xlim(0,350)
                #             axs[i, j].spines['top'].set_color('none')
                #             axs[i, j].spines['right'].set_color('none')
                #             axs[0, j].set_xlabel('Time (ms)', fontsize=15)
                #         axs[i, 3].set_ylabel('Frequency (Hz)', fontsize=15)
                #     plt.tight_layout()
                #     plt.savefig(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                #         args.learning_rate[lr]) + 'Mean_spectrogram_envelope.' + args.format)

                #
                # W and Delta W
                # W = np.load(args.data_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_W_sim_' + str(sim_counter) + '.npy')[0:args.time_limit, :, :]

                # Plot the evolution of the synaptic weights over trials
                # if np.size(args.T_names) == len(classes):
                #     fig, ax = plt.subplots(4, 4, sharex='col', sharey='row', figsize=(10, 5))
                #     for i in range(0, 4):
                #         for j in range(0, 4):
                #             for k in range(0, args.wavegan_latent_dim):
                #                 ax[i, j].plot(x_time, W[:, k, 4 * i + j], color[k])
                #             ax[i, j].set_ylabel('Weights', fontsize=8)
                #             ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                #             ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                #     plt.tight_layout()
                #     plt.savefig(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + 'Synaptic_weights_evolution_' + str(
                #             sim_counter) + '.' + args.format)

                # diff = []
                # for s in range(0, np.size(args.T_names)):
                   # diff.append(np.abs(np.diff(W[:, :, s], axis = 0)))

                # fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                # for i in range(0, 4):
                #     for j in range(0, 4):
                #         for w in range(0, args.wavegan_latent_dim):
                #             ax[i,j].plot(x_time[0:args.time_limit-1], diff[4 * i + j][:, w], 'b')
                #         ax[i, j].set_ylim(0, np.max(diff))
                #         ax[i,j].set_ylabel('Delta W', fontsize=8)
                #         ax[i,j].set_xlabel('Time (in number of time steps)', fontsize=8)
                #         ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                # plt.tight_layout()
                # plt.savefig(
                #     args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_diff_all' + str(sim_counter) + '.' + args.format)

                if np.size(args.T_names) == len(classes):
                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            ax[i, j].plot(x_time_expl, np.ones((np.shape(max_score_expl)[0], 1)), 'k')
                            ax[i, j].plot(x_time_expl, max_score_expl[:, 4 * i + j], 'b')
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_ylabel('Max score', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                    plt.tight_layout()
                    plt.savefig(
                        args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                            args.learning_rate[lr]) + '_max_score_expl' + str(
                            sim_counter) + '.' + args.format)

                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            ax[i, j].plot(x_time_expl, np.ones((np.shape(max_norm_expl)[0], 1)), 'k')
                            ax[i, j].plot(x_time_expl, max_norm_expl[:, 4 * i + j], 'b')
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_ylabel('Max-norm score', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                    plt.tight_layout()
                    plt.savefig(
                        args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                            args.learning_rate[lr]) + '_max_norm_expl' + str(
                            sim_counter) + '.' + args.format)

                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            ax[i, j].plot(x_time_expl, np.ones((np.shape(max_scaling_expl)[0], 1)), 'k')
                            ax[i, j].plot(x_time_expl, max_scaling_expl[:, 4 * i + j], 'b')
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_ylabel('Scaling score', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                    plt.tight_layout()
                    plt.savefig(
                        args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                            args.learning_rate[lr]) + '_max_scaling_expl' + str(
                            sim_counter) + '.' + args.format)

                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            ax[i, j].plot(x_time, np.ones((np.shape(max_score)[0], 1)), 'k')
                            ax[i, j].plot(x_time, max_score[:, 4 * i + j], 'b')
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_xlim(0, args.MAX_trial)
                            ax[i, j].set_ylabel('Max score', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                            ax[i, j].spines['top'].set_color('none')
                            ax[i, j].spines['right'].set_color('none')
                    plt.tight_layout()
                    plt.savefig(
                        args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                            args.learning_rate[lr]) + '_max_sim' + str(
                            sim_counter) + '.' + args.format)

                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            ax[i, j].plot(x_time, np.ones((np.shape(max_norm)[0], 1)), 'k')
                            ax[i, j].plot(x_time, max_norm[:, 4 * i + j], 'b')
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_xlim(0, args.MAX_trial)
                            ax[i, j].set_ylabel('Max-norm score', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                            ax[i, j].spines['top'].set_color('none')
                            ax[i, j].spines['right'].set_color('none')
                    plt.tight_layout()
                    plt.savefig(
                        args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                            args.learning_rate[lr]) + '_max_norm_sim' + str(
                            sim_counter) + '.' + args.format)

                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            ax[i, j].plot(x_time, np.ones((np.shape(max_scaling)[0], 1)), 'k')
                            ax[i, j].plot(x_time, max_scaling[:, 4 * i + j], 'b')
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_xlim(0, args.MAX_trial)
                            ax[i, j].set_ylabel('Scaling score', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                            ax[i, j].spines['top'].set_color('none')
                            ax[i, j].spines['right'].set_color('none')
                    plt.tight_layout()
                    plt.savefig(
                        args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                            args.learning_rate[lr]) + '_max_scaling_sim' + str(
                            sim_counter) + '.' + args.format)

                    # Sensory response raw score
                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            ax[i, j].plot(x_time, np.ones((np.shape(raw_score)[0], 1)), 'k')
                            ax[i, j].plot(x_time, raw_score[:, 4 * i + j], 'b')
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_xlim(0, args.MAX_trial)
                            ax[i, j].set_ylabel('Raw score', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                            ax[i, j].spines['top'].set_color('none')
                            ax[i, j].spines['right'].set_color('none')
                    plt.tight_layout()
                    plt.savefig(
                        args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                            args.learning_rate[lr]) + '_raw_score_sim' + str(
                            sim_counter) + '.' + args.format)

                    raw_score_sum = np.zeros((args.time_limit,))
                    for t in range(0, args.time_limit):
                        raw_score_sum[t] = np.sum(raw_score[t, :])

                    aux_save_raw = []
                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    print('Raw_score')
                    for i in range(0, 4):
                        for j in range(0, 4):
                            aux_save_raw.append(np.size(np.where(raw_score_expl[:, 4 * i + j] > 0.9)))
                            # print(np.size(np.where(raw_score_expl[:, 4 * i + j]>0.9)))
                            # input()
                            ax[i, j].plot(x_time_expl, np.ones((np.shape(raw_score_expl)[0], 1)), 'k')
                            ax[i, j].plot(x_time_expl, raw_score_expl[:, 4 * i + j], 'b')
                            ax[i, j].set_xlim(0, 300)
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_ylabel('Raw_score', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                            ax[i, j].spines['top'].set_color('none')
                            ax[i, j].spines['right'].set_color('none')
                    plt.tight_layout()
                    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                        args.learning_rate[lr]) + '_raw_score_expl' + str(
                        sim_counter) + '.' + args.format)

                    if args.learning_rate[lr] == 0.1:
                        np.save(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                            args.learning_rate[lr]) + '_cumulative_raw_score_expl.npy', aux_save_raw)

                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            h, bins = np.histogram(raw_score_expl[:, 4 * i + j], bins=15)
                            ax[i, j].bar(bins[:-1], h / np.max(h), width=0.05, color='b', alpha=0.6)
                            ax[i, j].spines['top'].set_color('none')
                            ax[i, j].spines['right'].set_color('none')
                            ax[i, j].set_xlim(0, 1)
                            ax[i, j].set_xlabel('Raw score', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                    plt.tight_layout()
                    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                        args.learning_rate[lr]) + '_raw_score_expl_hist' + str(sim_counter) + '.' + args.format)

                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            ax[i, j].plot(x_time, np.ones((np.shape(sensory_gen)[0], 1)), 'k')
                            ax[i, j].plot(x_time, sensory_gen[:, 4 * i + j], 'b')
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_ylabel('Soft-max', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                            ax[i, j].spines['top'].set_color('none')
                            ax[i, j].spines['right'].set_color('none')
                    plt.tight_layout()
                    plt.savefig(
                        args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr])+ '_Sensory_response_sim' + str(
                            sim_counter) + '.' + args.format)

                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            ax[i, j].plot(x_time_expl, np.ones((np.shape(sensory_expl_all)[0], 1)), 'k')
                            ax[i, j].plot(x_time_expl, sensory_expl_all[:, 4 * i + j], 'b')
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_ylabel('Soft-max', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                    plt.tight_layout()
                    plt.savefig(
                        args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(
                            args.learning_rate[lr]) + '_Sensory_response_expl_all' + str(
                            sim_counter) + '.' + args.format)

                    plt.close('all')

                    aux_save_softmax = []
                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    print('sensory_expl')
                    for i in range(0, 4):
                        for j in range(0, 4):
                            aux_save_softmax.append(np.size(np.where(sensory_expl[:, 4 * i + j] > 0.9)))
                            ##input()
                            ax[i, j].plot(x_time_expl, np.ones((np.shape(sensory_expl)[0], 1)), 'k')
                            ax[i, j].plot(x_time_expl, sensory_expl[:, 4 * i + j], 'b')
                            ax[i, j].set_xlim(0,300)
                            ax[i, j].set_ylim(0, 1)
                            ax[i, j].set_ylabel('Soft-max', fontsize=8)
                            ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                            ax[i, j].spines['top'].set_color('none')
                            ax[i, j].spines['right'].set_color('none')
                    plt.tight_layout()
                    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr])+ '_Sensory_response_expl' + str(sim_counter) + '.' + args.format)

                    if args.learning_rate[lr] == 0.1:
                        np.save(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr])+ '_cumulative_softmax_expl.npy', aux_save_softmax)

                    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
                    for i in range(0, 4):
                        for j in range(0, 4):
                            h, bins = np.histogram(sensory_expl[:, 4 * i + j], bins=15)
                            ax[i, j].bar(bins[:-1], h/np.max(h), width = 0.05, color='b', alpha=0.6)
                            ax[i, j].spines['top'].set_color('none')
                            ax[i, j].spines['right'].set_color('none')
                            ax[i, j].set_xlim(0, 1)
                            ax[i, j].set_xlabel('Soft-max', fontsize=8)
                            ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                    plt.tight_layout()
                    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr])+ '_Sensory_response_expl_hist' + str(sim_counter) + '.' + args.format)

            cfr_class_A_all.append(cfr_class_A)
            cfr_class_A_expl_all.append(cfr_class_A_expl)
            cfr_class_raw_all.append(cfr_class_raw)
            cfr_class_expl_all.append(cfr_class_expl)

        cfr_lr = ['10e-1', '10e-2']
        # CFR classifier sensory response
        for cl in range(0, len(args.classifier_name)):
            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    for lr in range(0, len(args.learning_rate)):
                        ax[i, j].plot(x_time, np.ones((np.shape(cfr_class_A_all[cl][lr])[0], 1)), 'k')
                        ax[i, j].plot(x_time, cfr_class_A_all[cl][lr][:, 4 * i + j], color=color[lr], label = cfr_lr[lr])
                    ax[i, j].set_ylim(0, 1)
                    ax[i, j].set_ylabel('Soft-max', fontsize=8)
                    ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                    ax[i, j].legend(loc='lower right', fontsize=5)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
            plt.tight_layout()
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_CFR_Sensory_response_sim' + str(sim_counter) + '.' + args.format)

            # CFR sensory response raw score
            fig, ax = plt.subplots(4, 4, figsize=(10, 5))
            for i in range(0, 4):
                for j in range(0, 4):
                    for lr in range(0, len(args.learning_rate)):
                        ax[i, j].plot(x_time, np.ones((np.shape(cfr_class_raw_all[cl][lr])[0], 1)), 'k')
                        ax[i, j].plot(x_time, cfr_class_raw_all[cl][lr][:, 4 * i + j], color=color[lr], label=cfr_lr[lr])
                    ax[i, j].set_ylim(0, 1)
                    ax[i, j].set_ylabel('Raw score', fontsize=8)
                    ax[i, j].set_xlabel('Time (in number of time steps)', fontsize=8)
                    ax[i, j].legend(loc='lower right', fontsize=5)
                    ax[i, j].set_title(classes[4 * i + j], fontsize=8)
                    ax[i, j].spines['top'].set_color('none')
                    ax[i, j].spines['right'].set_color('none')
            plt.tight_layout()
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[cl] + '_CFR_raw_score_sim' + str(
                    sim_counter) + '.' + args.format)

        # Ex syllable B
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), sharey=True, sharex=True)
        for lr in range(0, len(args.learning_rate)):
            axs[0].plot(x_time, np.ones((np.shape(cfr_class_expl_all[1][lr])[0], 1)), 'k')
            axs[0].plot(x_time, cfr_class_expl_all[1][lr][:, 1], 'b')
        axs[0].spines['top'].set_color('none')
        axs[0].spines['right'].set_color('none')
        axs[0].set_xlim(0, 300)
        #axs[0].set_xlabel('Time (in number of time steps)', fontsize=8)
        axs[0].legend(loc='lower right', fontsize=5)
        axs[0].set_ylabel('Raw score', fontsize=15)

        for lr in range(0, len(args.learning_rate)):
            axs[1].plot(x_time, np.ones((np.shape(cfr_class_raw_all[1][lr])[0], 1)), 'k')
            axs[1].plot(x_time, cfr_class_raw_all[1][lr][:, 1], color=color[lr], label = cfr_lr[lr])
        axs[1].spines['top'].set_color('none')
        axs[1].spines['right'].set_color('none')
        axs[1].set_xlim(0, 300)
        axs[1].set_xlabel('Time (in number of time steps)', fontsize=8)
        axs[1].legend(loc='lower right', fontsize=5)
        axs[1].set_ylabel('Raw score', fontsize=15)

        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'B1_realBIS.' + args.format)

        # Ex syllable C
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), sharey=True, sharex=True)
        for lr in range(0, len(args.learning_rate)):
            axs[0].plot(x_time, np.ones((np.shape(cfr_class_expl_all[0][lr])[0], 3)), 'k')
            axs[0].plot(x_time, cfr_class_expl_all[0][lr][:, 3], 'b')
        axs[0].set_xlim(0, 300)
        axs[0].spines['top'].set_color('none')
        axs[0].spines['right'].set_color('none')
        #axs[0].set_xlabel('Time (in number of time steps)', fontsize=8)
        axs[0].legend(loc='lower right', fontsize=5)
        axs[0].set_ylabel('Raw score', fontsize=15)

        for lr in range(0, len(args.learning_rate)):
            axs[1].plot(x_time, np.ones((np.shape(cfr_class_raw_all[0][lr])[0], 3)), 'k')
            axs[1].plot(x_time, cfr_class_raw_all[0][lr][:, 3], color=color[lr], label = cfr_lr[lr])
        axs[1].set_xlim(0, 300)
        axs[1].spines['top'].set_color('none')
        axs[1].spines['right'].set_color('none')
        axs[1].set_xlabel('Time (in number of time steps)', fontsize=8)
        axs[1].legend(loc='lower right', fontsize=5)
        axs[1].set_ylabel('Raw score', fontsize=15)

        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'C_extBIS.' + args.format)

        input()

        if np.size(args.T_names) == 3:
            fig, ax = plt.subplots(args.ns, 1, figsize=(5, 10))
            for j in range(0, args.ns):
                ax.flat[j].plot(x_time, np.ones((np.shape(sensory_gen)[0], 1)))
                ax.flat[j].plot(x_time, sensory_gen[:,j], color[j], label='Syllable '+ args.T_names[j])
                ax[j].set_ylabel('Sensory response', fontsize=15)
                ax[j].set_xlabel('Time (in number of time steps)', fontsize=15)
            plt.legend(loc='lower right', fontsize=15)
            plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'Sensory_response_sim' + str(sim_counter) + '.' + args.format)

            fig, ax = plt.subplots(args.ns, 1, figsize=(5, 10))
            for j in range(0, args.ns):
                ax.flat[j].plot(x_time, np.ones((np.shape(sensory_expl)[0], 1)))
                ax.flat[j].plot(x_time, sensory_expl[:, j], color[j], label='Syllable ' + args.T_names[j])
                ax[j].set_ylabel('Sensory response', fontsize=15)
                ax[j].set_xlabel('Time (in number of time steps)', fontsize=15)
            plt.legend(loc='lower right', fontsize=15)
            plt.savefig(
                args.data_dir + '/' + args.output_dir + '/' + 'Sensory_response_expl' + str(sim_counter) + '.' + args.format)

    print('Done')

def plot_syll(args):
    """
    Plot the example of a syllable across time: change the name in syllables variable (just below this comment)
    """
    syllables = glob.glob(args.data_dir + '/' + '*R.wav')
    counter = 0
    while counter < len(syllables):
        samples_aux, sr = librosa.load(syllables[counter], sr=16000)
        trim = librosa.effects.trim(samples_aux.astype(np.float), top_db=20)
        samples_aux = trim[0]

        X = librosa.stft(samples_aux, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
        Y = np.log(1 + 100 * np.abs(X) ** 2)
        T_coef = np.arange(X.shape[1]) * args.H / sr
        K = args.N // 2
        F_coef = np.arange(K + 1) * sr / args.N

        plt.figure(figsize=(4, 18))
        extent = [T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
        plt.imshow(Y, aspect='auto', origin='lower', extent=extent, cmap=args.color, norm=colors.PowerNorm(gamma=0.5))
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title(str(counter))
        plt.tight_layout()
        plt.savefig(args.data_dir + '/' + args.output_dir + '/' + 'R' + '.' + args.format)

        counter = counter + 1
    print('Done')

def mean_spectro(learning_rate, sim_counter, ths, args):
    """
    
    :param learning_rate: which learning rate
    :param sim_counter: which simulation
    :param ths threshold to define activation
    
    :return: mean spectogram for each syllable when it is active more than a threshold
    """
    # Load activation function and list of directories
    p95 = np.load(args.data_dir + '/' + args.classifier_name[0] + '_lr' + str(learning_rate) + '_p95_sim_' + str(sim_counter) + '.npy')

    # Remove focus (every N points up to 200 points)
    p95_begin = p95[0:200, :]
    p95_jump = np.zeros((args.n_points + 1, np.size(args.T_names)))
    p95_jump[0:14, :] = p95_begin[0::15, :]
    p95_jump[14::, :] = p95[200::, :]

    list = np.zeros((args.n_points + 1,))
    aux = np.linspace(0, 3000, 3000).astype(int)
    list[0:200] = aux[0::15]
    list[-1] = 3000

    mean_spectrogram_env = []
    T = []
    for c in range(0, np.size(args.T_names)):
        # Find where the activation threshold is reached/crossed
        loc = np.where(p95_jump[:, c] > ths)[0]

        spectrograms_envelope = []

        for sp in range(0, np.size(loc)):
            if loc[sp] < 200:
                samples_aux, sr = librosa.load(
                    args.data_dir + '/' + args.sim_name + str(sim_counter) + '/' + args.classifier_name[
                        0] + '_lr' + str(learning_rate) + '_' + args.sim_name + str(sim_counter) + '_' + str(
                        int(list[loc[sp]])) + '/' + '__condition_0_' + str(int(list[loc[sp]]))  + '/' + 'sensory_production_condition_0_' + args.T_names[c] + '.wav', sr=16000)
            else:
                loc[sp] = loc[sp]
                samples_aux, sr = librosa.load(
                    args.data_dir + '/' + args.sim_name + str(sim_counter) + '/' + args.classifier_name[
                        0] + '_lr' + str(learning_rate) + '_' + args.sim_name + str(sim_counter) + '_' + str(
                        int(list[loc[sp]])) + '/' + '__condition_0_' + str(int(list[loc[sp]]))  + '/' + 'sensory_production_condition_0_' + args.T_names[c] + '.wav', sr=16000)
            trim = librosa.effects.trim(samples_aux.astype(np.float), top_db=20)
            samples_aux = trim[0]

            if samples_aux.size / 16 < 4000:
                aux_size = 4000 - samples_aux.size / 16
                silence = np.zeros((int(round(aux_size / 2) * 16)), )
                samples_aux = np.append(silence, samples_aux)
                samples_aux = np.append(samples_aux, silence)

            rawsong = samples_aux.astype(float)
            rawsong = rawsong.flatten()
            amp = Song_functions.smooth_data(rawsong, sr, freq_cutoffs=(500, 7999))

            new_song = rawsong[np.where(amp > 0.00001)[0][0]::]
            silence = np.zeros((50000 - np.size(new_song),))
            new_song = np.append(new_song, silence)

            X = librosa.stft(new_song, n_fft=args.N, hop_length=args.H, win_length=args.N, window='hann', pad_mode='constant', center=True)
            T_coef = np.arange(X.shape[1]) * args.H / sr * 1000
            spectrograms_envelope.append(np.log(1 + 100 * np.abs(X ** 2)))

        mean_spectrogram_env.append(np.mean(spectrograms_envelope, axis=0))  # dimension 16
        T.append(T_coef)

    #np.save(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[0] + '_' + str(sim_counter) + '_lr' + str(args.learning_rate) + 'Mean_spectrogram_envelope', mean_spectrogram_env)

    # Mean spectrogram after convergence (plot)
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 14), sharey=True, sharex=True)
    for i in range(0, 4):
        for j in range(0, 4):
            extent = [0, 300, 0, 8000]
            if mean_spectrogram_env[4 * i + j].size > 1:
                axs[i, j].imshow(mean_spectrogram_env[4 * i + j], extent=extent, cmap=args.color, aspect='auto', origin='lower', norm=colors.PowerNorm(gamma=0.5))  # gamma 0.2 in original data
            axs[i, j].set_title(args.T_names[4 * i + j], fontsize=15)
            axs[i, j].set_xlim(0,20)
            axs[i, j].spines['top'].set_color('none')
            axs[i, j].spines['right'].set_color('none')
            axs[0, j].set_xlabel('Time (ms)', fontsize=15)
        axs[i, 3].set_ylabel('Frequency (Hz)', fontsize=15)
    plt.tight_layout()
    plt.savefig(args.data_dir + '/' + args.output_dir + '/' + args.classifier_name[0] + '_' + str(sim_counter) + '_lr' + str(args.learning_rate) + 'Mean_spectrogram_envelope.' + args.format)

    print('Done')

if __name__ == '__main__':
    import argparse
    import glob
    import sys
    """
    Example how to run it:
    >python plotGAN.py --option learning --data_dir experiment --output_dir plots
    
    The output_dir will be created by default inside the data directory.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--option', type=str,
                        help='What do you want to see? Motor exploration or results after learning?',
                        choices=['sensory', 'activation_aud', 'syll', 'mean_spectro', 'cfr'])
    parser.add_argument('--data_dir', type=str,
                        help='Data directory where the data are saved',
                        default=None)
    parser.add_argument('--output_dir', type=str,
                        help='Output directory where to save the plots',
                        default=None)

    simulation_args = parser.add_argument_group('Simulation')
    simulation_args.add_argument('--MAX_trial', type=int,
                                 help='Maximal number of trials',
                                 default = 3001)
    simulation_args.add_argument('--ns', type=int,
                                 help='number of syllables',
                                 default = 16)
    simulation_args.add_argument('--N_sim', type=int, help='Number of instances', default=3)
    simulation_args.add_argument('--T_names', type=list, help='Target syllables', default=['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']) #['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']) #['B1', 'C', 'M'])
    simulation_args.add_argument('--sim_name', type=str, help='Sub directory containing the generations per each simulation', default='sensory_prod_sim_')
    simulation_args.add_argument('--classifier_name', type=list, help='Which classifier model I want to use. Multiple classifier are allowed', default=['EXT']) #'REAL'
    simulation_args.add_argument('--learning_rate', type=list,
                         help='Learning rate used during learning',
                         default = [0.1, 0.01]) #[0.1, 0.01]
    simulation_args.add_argument('--beta', type=list, help='Type of auditory softmax activation',
                                 default=[0.01, 0.1, 1, 5])

    spectro_args = parser. add_argument_group('Spectorgram')
    spectro_args.add_argument('--N', type = int, help='Nftt spectrogram librosa', default=256)
    spectro_args.add_argument('--H', type = int, help='Hop length spectrogram librosa', default=64)
    spectro_args.add_argument('--color', type = str, help='Colormap', default='inferno')

    # TODO add reading of the params file, it could be that I need to change in the InverseLearningGAN the way I save
    # args.txt. Perhaps using a dict or json instead or in addition.

    wavegan_args = parser.add_argument_group('WaveGAN')
    wavegan_args.add_argument('--wavegan_latent_dim', type=int,
                              help='Dimension of the latent space',
                              default = 2)

    plot_args = parser.add_argument_group('Plots')
    plot_args.add_argument('--format', type=str, help='Saving format', default='png')
    plot_args.add_argument('--time_limit', type=int, help='Print only a certain time', default=100)
    plot_args.add_argument('--n_points', type=int, help='How many point to be plot in the figure (=to saved points)', default=200)
    plot_args.add_argument('--example', type=str, help='Figure of an example', default=True)

    args = parser.parse_args()

    # Make output dir
    if args.output_dir != None:
        if not os.path.isdir(args.data_dir + '/' + args.output_dir):
            os.makedirs(args.data_dir + '/' + args.output_dir)

    if args.option == 'activation_aud':
        plot_auditory_activation(args)

    if args.option == 'sensory':
        plot_sensory(args)

    if args.option == 'syll':
        plot_syll(args)

    if args.option =='mean_spectro':
        learning_rate = 0.01
        ths = 0.99
        sim_counter = 2
        mean_spectro(learning_rate, sim_counter, ths, args)

    if args.option =='cfr':
        # Latent space conditions
        ld = [1, 2, 3, 6]
        colors = ['r', 'b', 'gold', 'k']

        p95_MEAN =[]
        for i in range(0,len(ld)):
            p95_MEAN.append(np.load(args.data_dir + '/' + 'p95_MEAN_lr_' + str(ld[i]) + '.npy'))

        cfr_dim13(p95_MEAN, colors, ld, args)