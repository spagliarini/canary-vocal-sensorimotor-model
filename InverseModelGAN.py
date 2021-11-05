# -*- coding: utf-8 -*-
"""
Created on Tue 17 March 18:31:55 2020

@author: Mnemosyne

Vocal learning model
- target space: perceptual activation at 1 (perceptual target)
- exploration space: latent space from the WaveGAN generation after training
- sensory function: classifier (last one, with 5 garnage classes, called classifier-EXT)
- perception space: classifier + activation function (using the global 95-percentile)
- learning architecture: inverse model, Hebbian learning rule to update the learning weights
- motor space: has the same dimension of the latent space used to train WaveGAN
- motor function: generator from WaveGAN
- sensory space: sound generated after having updated the learning weights
"""

import os
import time
import glob
import pickle
import random
import librosa
import xlsxwriter
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import rcParams, cm, colors
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import numpy as np
import scipy as sp
import scipy.special
import scipy.io.wavfile as wav
from random import randrange
from shutil import copyfile

# Auxiliary function to open pkl file
def open_pkl(name):
    data = open(name, 'rb')
    z = pickle.load(data)
    return z

def softmax_beta(beta, z):
    """
    Function to compute the softmax with the possibility f varying the parameter beta.
    The scipy.special.softmax function is this function with beta = 1 .

    :param beta: parameter of softmax
    :param z: vector to apply the softmax to
    :return: softmax (vector)
    """
    softmax = np.exp(np.dot(beta,z))/np.sum(np.exp(np.dot(beta,z)))

    return softmax

def target(args):
    '''
    Function to select the target (ideally the syllables that activate the most the classifier).
    '''
    lv = glob.glob('generation_' + str(args.ckpt_n) + '/' + 'z*.pkl')
    annotations = open_pkl(args.template_dir + '/' + 'annotations.pkl')
    classes = annotations[0].vocab

    np.save(args.template_dir + '/' + 'vocab.npy', classes)

    raw_sum = np.zeros((len(annotations), np.size(classes)))
    raw_sum_distr = np.zeros((len(annotations), np.size(classes)))
    raw_max = np.zeros((len(annotations), ))
    raw_max_indices = np.zeros((len(annotations), ))
    annotations_raw = []
    annotations_id = []
    for i in range(0,len(annotations)):
        annotations_raw.append(annotations[i].vect)
        annotations_id.append(annotations[i].id)
        # This operation should give me an idea of which class is represented the most in my generations
        raw_sum[i, :] = np.sum(annotations_raw[i], axis=0)
        # Probability distribution to compute the inception score
        raw_sum_distr[i, :] = sp.special.softmax(raw_sum[i, :])
        raw_max[i] = np.max(raw_sum_distr[i, :])
        raw_max_indices[i] = np.where(raw_sum_distr[i, :] == np.max(raw_sum_distr[i, :]))[0][0]

    # Select the best one
    best_gen = []
    best_lv = []
    best_vect = []
    mean = []
    min =[]
    max = []
    for c in range(0,np.size(classes)):
        aux_index_list = np.where(raw_max_indices == c)
        aux_vect = raw_max[aux_index_list]
        aux_target = np.max(aux_vect)

        mean.append(np.mean(raw_max[aux_index_list]))
        min.append(np.min(raw_max[aux_index_list]))
        max.append(np.max(raw_max[aux_index_list]))

        aux_best = aux_index_list[0][np.where(aux_vect == aux_target)[0][0]]
        if aux_best.size>1:
           aux_best = aux_best[0]

        best_gen.append(classes[c] + ': ' + os.path.basename(annotations_id[aux_best]))
        best_vect.append(raw_sum_distr[aux_best,:])

        # latent vector
        #best_lv.append(os.path.basename(lv[aux_best]))
        #print(lv[aux_best])

        #copyfile(lv[aux_best], args.template_dir + 'z_' + classes[c] + '.pkl')

    np.save(args.template_dir + '/' + 'perceptual_pattern.npy', best_vect)

    # Initialize sheet
    workbook = xlsxwriter.Workbook(args.template_dir + '/' + 'Summary_table_target.xlsx')
    worksheet = workbook.add_worksheet()

    # Start from the first cell.
    content = ["Class", "Min", "Mean", "Max", "Best", "z"]
    # Rows and columns are zero indexed.
    row = 0
    column = 0
    # iterating through content list
    for item in content:
        # write operation perform
        worksheet.write(row, column, item)

        # incrementing the value of row by one
        # with each iteratons.
        column += 1

    row = 1
    column = 0
    for item in range(0, np.size(classes)):
        # write test names
        worksheet.write(row, column, classes[item])

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    row = 2
    column = 1
    for item in range(0, len(min)):
        # write test names
        worksheet.write(row, column, min[item])

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    row = 2
    column = 2
    for item in range(0, len(mean)):
        # write test names
        worksheet.write(row, column, mean[item])

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    row = 2
    column = 3
    for item in range(0, len(max)):
        # write test names
        worksheet.write(row, column, max[item])

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    row = 2
    column = 4
    for item in range(0, len(best_gen)):
        # write test names
        worksheet.write(row, column, best_gen[item])

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    row = 2
    column = 5
    for item in range(0, len(best_lv)):
        # write test names
        worksheet.write(row, column, best_lv[item])

        # incrementing the value of row by one
        # with each iteratons.
        row += 1

    workbook.close()

    print('Done')

# Learning algorithm
# Classic Hebbian learning rule
def IM_simple_classic(eta,M,A,W, ns, ld):
    """
    params:
        - eta: learning rate
        - M: motor exploration
        - A: auditory activity
        - W: current weights matrix
    """
    A = np.asarray(A.reshape(ns,1))
    M = np.asarray(M.reshape(ld,1))

    DeltaW = eta * np.dot(M,A.T)
    W = W + DeltaW

    return W, DeltaW

# Motor function
def motor_function_WaveGAN(train_dir, latent_vector, filename, sampling_rate, output_dir):
    '''
    This function generates the sound starting from a vector having the same dimension of the latent space:
    it has to be fixed before to start the WaveGAN training.

    To use this function the train directory of the model needs to be available in order to get access to the
    checkpoint.

    It returns one vocal generation: for example, if I want to associate on vector to one syllable than this
    function takes as input one vector (saved as pkl) and generates one syllable.

    Partially taken from train_wavegan.py in WaveGAN
    '''
    # Load the graph
    tf.reset_default_graph()
    infer_metagraph_fp = os.path.join(train_dir, 'infer', 'infer.meta')
    saver = tf.train.import_meta_graph(infer_metagraph_fp)
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()

    if args.ckpt_n == False:
        ckpt_fp = tf.train.latest_checkpoint(train_dir)
    else:
        ckpt_fp = os.path.join(train_dir, 'model.ckpt-'+str(args.ckpt_n))
    saver.restore(sess, ckpt_fp)

    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]

    _G_z = sess.run(G_z, {z: latent_vector})

    #wav.write(output_dir + '/' + filename, sampling_rate, _G_z.T)
    librosa.output.write_wav(output_dir + '/' + filename, _G_z.T, sampling_rate)

    sess.close()

    return

# Sensory response function
def sensory_response(dir):
    """
    The input directory contains one or more audio files (.wav) of duration 1s.

    This function creates a dictionary containing three elements:
    - mean: this entry strores the averaged outputs produced by the ESN, for each sample. This gives each sample a unique annotation vector;
            the mean output link the whole output (the whole syllable) to a 16 components vector (which is an indicator vector for the 16 classes of syllables)
    - raw: this entry stores the raw outputs produced by the ESN. The ESN produce one annotation vector per timestep;
           the raw output does the same thing but for each timestep of input audio
    - states: this entry stores the internal states of the ESN over time.

    The output is saved in the same directory of the data (outside to be able to use the function somewhere else).
    """

    from canarydecoder import load

    # Load the model (change depending on which we want to use - usually REAL)
    # REAL
    decoder = load('canary16-filtered-notrim')
    # EXT (5 garbage classes)
    # decoder = load('canarygan-f-3e-ot-noise-notrim')

    # Create dictionary
    annotations = decoder(dir)

    return annotations

def auditory_activation(option, annotations, classes, vocabulary, scaling_wo):
    # Raw score
    if option == 'raw_score':
        raw_sum = np.sum(annotations, axis=0)
        raw_score_aux = np.zeros((np.size(classes),))
        for s in range(0, np.size(classes)):
            raw_score_aux[s] = (raw_sum[np.where(vocabulary == classes[s])] - np.min(raw_sum)) / (
                        np.max(raw_sum) - np.min(raw_sum))

        return raw_score_aux

    # Probability distribution with soft max
    if option == 'softmax':
        raw_sum = np.sum(annotations, axis=0)
        A_all = scipy.special.softmax(raw_sum)
        A = np.zeros((np.size(classes),))
        for s in range(0, np.size(classes)):
            A[s] = A_all[np.where(vocabulary == classes[s])]

        return A_all

    # Max activity in half a second, scaling
    if option == 'max':
        annotations_raw = annotations[0:16, :]
        MAX = np.max(np.max(annotations_raw, axis=0))  # Find the max for each class, and take the max of the max
        index_MAX = np.where(np.max(annotations_raw, axis=0) == np.max(
            np.max(annotations_raw, axis=0)))  # Where is the max in the annotations, in which syllable
        index_TIME = np.where(annotations_raw[:, index_MAX] == MAX)[0][
            0]  # Where is the max in time, for a given syllables
        column_TIME = annotations_raw[index_TIME,
                      :]  # Remove values less than zero and bigger than one (piecewise function)
        for i in range(0, np.size(vocabulary)):
            if column_TIME[i] < 0.01:
                column_TIME[i] = 0
            if column_TIME[i] > 1:
                column_TIME[i] = 1
        max_expl_aux = np.zeros((np.size(args.T_names),))
        max_norm_expl_aux = np.zeros((np.size(args.T_names),))
        scaling_expl_wo_aux = np.zeros((np.size(classes),))
        for s in range(0, np.size(args.T_names)):
            max_expl_aux[s] = column_TIME[np.where(vocabulary == args.T_names[s])]
            max_norm_expl_aux[s] = column_TIME[np.where(vocabulary == args.T_names[s])] / np.max(column_TIME)
            scaling_expl_wo_aux[s] = (column_TIME[np.where(vocabulary == classes[s])] - scaling_wo[s][0]) / (scaling_wo[s][1] - scaling_wo[s][0])

        return max_expl_aux, max_norm_expl_aux, scaling_expl_wo_aux

def pre_def(args):
    # Initialization of the weights
    W_in = np.random.uniform(-args.W_seed, args.W_seed, (args.wavegan_latent_dim, np.size(args.T_names)))
    np.save(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy', W_in)

    # Motor exploration: pick a random motor exploration
    all_motor = sorted(glob.glob(args.exploration_dir + '/' + '*.pkl'))
    expl = np.random.choice(range(np.size(all_motor)), args.MAX_trial)
    np.save(args.output_dir + '/' + 'motor_exploration_' + str(args.wavegan_latent_dim) + '.npy', expl)

    print('Done')

def exploration_space(args):
    # Load the annotations
    annotations_aux = []
    vocabulary = []
    for cl in range(0, len(args.classifier_name)):
        annotations_aux.append(open_pkl(args.sensory_dir + '/' + 'sensory_' + str(args.wavegan_latent_dim) + '_' + args.classifier_name[cl] + '.pkl'))
        vocabulary.append(annotations_aux[cl][0].vocab)

        # GAN classes (needed if classifier-EXT is used)
        if args.classifier_name[cl] != 'REAL':
            GAN_class = np.zeros((np.size(args.GAN_classes),))
            for gc in range(0, np.size(args.GAN_classes)):
                GAN_class[gc] = np.where(vocabulary[cl] == args.GAN_classes[gc])[0][0]

    # Classes
    classes = ['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V', 'X']

    # Color map
    classes_colors = ['springgreen', 'firebrick', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen', 'tan', 'darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'k']
    #classes_colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'limegreen', 'darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'indigo', 'violet', 'deeppink', 'white']
    classes_cmap = colors.LinearSegmentedColormap.from_list("14classes_garbage", classes_colors)
    plt.register_cmap("14classes_garbage", classes_cmap)

    # Scaling to compute the score in the auditory activation
    max_annotations_aux = []
    for i in range(0, len(annotations_aux[0])):
        # Read the annotations
        annotations_raw = annotations_aux[0][i].vect

        # Consider only ~500 ms of sound (after is silence)
        annotations_raw = annotations_raw[0:16,:]

        # Find the max for each class, and take the max of the max
        MAX = np.max(np.max(annotations_raw, axis=0))

        # Where is the max in the annotations, in which syllable
        index_MAX = np.where(np.max(annotations_raw, axis = 0) == np.max(np.max(annotations_raw, axis = 0)))

        # Where is the max in time, for a given syllables
        index_TIME = np.where(annotations_raw[:, index_MAX] == MAX)[0][0]

        # Remove values less than zero and biggger than one (piecewise function)
        column_TIME = annotations_raw[index_TIME, :]

        for i in range(0, np.size(vocabulary)):
            if column_TIME[i]<0.01:
                column_TIME[i] = 0
            if column_TIME[i]>1:
                column_TIME[i] = 1

        # Consider the column where MAX is located
        max_annotations_aux.append(column_TIME)

    max_C = np.max(max_annotations_aux, axis=0)
    min_C = np.min(max_annotations_aux, axis=0)
    p95 = np.percentile(max_annotations_aux, 95, axis=0)
    scaling = np.zeros((np.size(vocabulary),3))

    for c in range(0, np.size(vocabulary)):
        scaling[c,0] = min_C[c]
        scaling[c,1] = max_C[c] - min_C[c]
    scaling[:,2] = p95[:]
    print(scaling)
    print(vocabulary[0])

    np.save(args.output_dir + '/' + 'scaling_' + str(args.wavegan_latent_dim) + '_' + args.classifier_name[0] + '.npy', scaling)

    # Activation based on the precomputed p95 score for each element of the exploration space
    decoder_name_gen = []
    for i in range(0, len(annotations_aux[0])):
        # Max vector and score
        aux_act = np.zeros((np.size(vocabulary),))
        for s in range(0, np.size(vocabulary)):
            aux_act[s] = max_annotations_aux[i][s]/scaling[s,2]
            if aux_act[s]>1:
                aux_act[s]=1

        # Which is the active syllable
        max_act = np.max(aux_act)
        index_max = np.where(aux_act == max_act)[0][0]
        decoder_name_gen.append(vocabulary[0][index_max])

    # Latent space representation
    load_summary = np.load(args.sensory_dir + '/' + 'sensory_' + str(args.wavegan_latent_dim) + '_' + args.classifier_name[cl] + '.npy', allow_pickle=True)
    load_summary = load_summary.item()
    file_name_gen = load_summary['File_name']

    if args.wavegan_latent_dim == 3:
        data_dir = 'D:\PhD_Bordeaux\Python_Scripts\Generative_Models\wavegan-master\GPUbox_Results\FILTtrainPLAFRIM_16s_Marron1_ld3sette\Gan16000'
        xy = np.zeros((len(file_name_gen), 2))
        xz = np.zeros((len(file_name_gen), 2))
        yz = np.zeros((len(file_name_gen), 2))
        z = np.zeros((len(file_name_gen), 3))
        c = []
        for j in range(0, len(file_name_gen)):
            latent_aux_path = data_dir + '/' + 'generation_49202' + '/' + 'z' + os.path.basename(file_name_gen[j])[10:-4] + '.pkl'
            z_aux = open_pkl(latent_aux_path)[0]

            z[j, :] = z_aux
            xy[j, :] = [z_aux[0], z_aux[1]]
            xz[j, :] = [z_aux[0], z_aux[2]]
            yz[j, :] = [z_aux[1], z_aux[2]]

            # Decoder name additional X class and grouping J and B
            if decoder_name_gen[j] == 'EARLY15':
                decoder_name_gen[j] = 'X'
            if decoder_name_gen[j] == 'EARLY30':
                decoder_name_gen[j] = 'X'
            if decoder_name_gen[j] == 'EARLY45':
                decoder_name_gen[j] = 'X'
            if decoder_name_gen[j] == 'OT':
                decoder_name_gen[j] = 'X'
            if decoder_name_gen[j] == 'WN':
                decoder_name_gen[j] = 'X'

            c.append(classes_colors[np.where(np.asarray(classes) == decoder_name_gen[j])[0][0]])

        # How many syllable per type
        how_many = np.zeros((len(classes),))
        for k in range(0, len(classes)):
            how_many[k] = np.size(np.where(np.asarray(decoder_name_gen) == classes[k])[0])
        print(how_many)
        print(how_many*100/16000)

        # Creation of the legend
        legend_elements = []
        for i in range(0, len(classes)):
            legend_elements.append(Line2D([0], [0], marker='o', color=classes_colors[i], label=classes[i]))

        #plt.style.use('dark_background')

        # Plot
        # 3D slices
        bound = -1
        step = 0.25
        while bound < 0.85:
            aux = np.where(np.logical_and(z[:, 1] >= bound, z[:, 1] <= bound + step))[0]

            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(z[aux, 0], z[aux, 1], z[aux, 2], c=np.asarray(c)[aux], marker='.')
            ax.set_xlabel('z0')
            ax.set_ylabel('z1')
            ax.set_zlabel('z2')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.legend(handles=legend_elements)
            plt.tight_layout()
            if args.format != 'png':
                plt.savefig(args.output_dir + '/' + 'slice_' + str(
                    bound) + '_latent_z.' + args.format)
            plt.savefig(args.output_dir + '/' + 'slice_' + str(bound) + '_latent_z.' + 'png')

            bound = bound + step

    elif args.wavegan_latent_dim == 1:
        data_dir = 'D:\PhD_Bordeaux\Python_Scripts\Generative_Models\wavegan-master\GPUbox_Results\FILTtrainPLAFRIM_16s_Marron1_ld1\Gen16000'
        z = np.zeros((len(file_name_gen),))
        c = []
        for j in range(0, len(file_name_gen)):
            latent_aux_path = data_dir + '/' + 'generation_50002' + '/' + 'z' + os.path.basename(file_name_gen[j])[10:-4] + '.pkl'
            z_aux = open_pkl(latent_aux_path)[0]

            z[j] = z_aux

            # Decoder name additional X class and grouping J and B
            if decoder_name_gen[j] == 'EARLY15':
                decoder_name_gen[j] = 'X'
            if decoder_name_gen[j] == 'EARLY30':
                decoder_name_gen[j] = 'X'
            if decoder_name_gen[j] == 'EARLY45':
                decoder_name_gen[j] = 'X'
            if decoder_name_gen[j] == 'OT':
                decoder_name_gen[j] = 'X'
            if decoder_name_gen[j] == 'WN':
                decoder_name_gen[j] = 'X'

            c.append(classes_colors[np.where(np.asarray(classes) == decoder_name_gen[j])[0][0]])

        # How many syllable per type
        how_many = np.zeros((len(classes),))
        for k in range(0, len(classes)):
            how_many[k] = np.size(np.where(np.asarray(decoder_name_gen) == classes[k])[0])
        print(how_many)
        print(how_many * 100 / 16000)

        # Creation of the legend
        legend_elements = []
        for i in range(0, len(classes)):
            legend_elements.append(Line2D([0], [0], marker='o', color=classes_colors[i], label=classes[i]))

        #plt.style.use('dark_background')

        # Plot
        # Line plot
        bound = -1
        step = 0.25
        while bound < 0.85:
            aux = np.where(np.logical_and(z[:] >= bound, z[:] <= bound + step))[0]

            plt.subplots(figsize=(4,1))
            for i in range(0, np.size(aux)):
                plt.plot(z[aux[i]], np.zeros_like(z[aux[i]]) + 0, c=np.asarray(c)[aux[i]], marker='.')
            plt.tight_layout()
            plt.xlabel('z0')
            plt.yticks([])
            if args.format != 'png':
                plt.savefig(args.output_dir + '/' + str(
                    bound) + '_latent_z.' + args.format)
            plt.savefig(args.output_dir + '/' + str(
                bound) + '_latent_z.' + 'png')

            bound = bound + step

        input()
        bound = 0
        step = 0.25
        while bound < 0.85:
            aux = np.where(np.logical_and(z[:] >= bound, z[:] <= bound + step))

            plt.figure()
            ax = plt.axes()
            ax.plot(z[aux], c=np.asarray(c)[aux], marker='.')
            ax.set_xlabel('z0')
            plt.tight_layout()
            if args.format != 'png':
                plt.savefig(args.output_dir + '/' + str(
                    bound) + '_latent_z.' + args.format)
            plt.savefig(args.output_dir + '/' + str(
                    bound) + '_latent_z.' + 'png')

            bound = bound + step

        # 1D scatter plot
        aux = np.where(z[:] <=0)
        plt.figure()
        ax = plt.axes()
        ax.scatter(z[aux], z[aux], c=np.asarray(c)[aux], marker='.')
        ax.set_xlabel('z0')
        ax.set_xlim([-1, 0])
        ax.set_ylabel('z0')
        ax.set_ylim([-1, 0])
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        if args.format != 'png':
            plt.savefig(args.output_dir + '/' + 'latent_z_neg.' + args.format)
        plt.savefig(args.output_dir + '/' + 'latent_z_neg.' + 'png')

        aux = np.where(z[:] >= 0)
        plt.figure()
        ax = plt.axes()
        ax.scatter(z[aux], z[aux], c=np.asarray(c)[aux], marker='.')
        ax.set_xlabel('z0')
        ax.set_xlim([0, 1])
        ax.set_ylabel('z0')
        ax.set_ylim([0, 1])
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        if args.format != 'png':
            plt.savefig(args.output_dir + '/' + 'latent_z_pos.' + args.format)
        plt.savefig(args.output_dir + '/' + 'latent_z_pos.' + 'png')

    plt.close('all)')

    print('Done')

def auditory_activation_test(args):
    all_motor = sorted(glob.glob(args.exploration_dir + '/' + '*.pkl'))

    annotations_aux = []
    vocabulary = []
    scaling = []
    scaling_wo = []
    for cl in range(0, len(args.classifier_name)):
        annotations_aux.append(open_pkl(args.sensory_dir + '/' + 'sensory_' + str(args.wavegan_latent_dim) + '_' + args.classifier_name[cl] + '.pkl'))
        vocabulary.append(annotations_aux[cl][0].vocab)

        scaling_aux = np.load(args.sensory_dir + '/' + 'scaling_' + str(args.wavegan_latent_dim) + '_' + args.classifier_name[cl] + '.npy')
        scaling_wo_aux = np.zeros((np.size(args.T_names),3))
        for s in range(0, np.size(args.T_names)):
            scaling_wo_aux[s,:] = scaling_aux[np.where(vocabulary[cl] == args.T_names[s]),:]
        scaling.append(scaling_aux)
        scaling_wo.append(scaling_wo_aux)

        # GAN classes (needed if classifier-EXT is used)
        if args.classifier_name[cl] != 'REAL':
            GAN_class = np.zeros((np.size(args.GAN_classes),))
            for gc in range(0, np.size(args.GAN_classes)):
                GAN_class[gc] = np.where(vocabulary[cl] == args.GAN_classes[gc])[0][0]

    for sim_counter in range(0, args.N_sim):
        # Motor exploration: pick a random motor exploration
        expl = np.random.choice(range(np.size(all_motor)), args.MAX_trial, replace=True)
        np.save(args.output_dir + '/' + 'motor_exploration_sim_' + str(sim_counter) + '.npy', expl)

        raw_score_expl = []
        softmax_sum_expl = []
        softmax_mean_expl = []
        mean_norm_expl = []
        mean = []
        logistic_expl = []
        minmax_expl = []
        sign_minmax_expl = []
        tanh_expl = []
        sign_expl = []
        square_expl = []
        arctg_expl = []
        scaling_expl = []
        softmax_MAX_expl = []
        scaling_softmax_expl = []
        max_expl = []
        max_norm_expl = []
        p95_expl = []

        trial_nr = 0
        while trial_nr < args.MAX_trial:
            # Sensory function (classifier)
            annotations_raw = annotations_aux[cl][expl[trial_nr]].vect
            vocab = annotations_aux[cl][expl[trial_nr]].vocab

            raw_mean = np.mean(annotations_raw, axis=0)
            mean_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                mean_aux[s] = raw_mean[np.where(vocabulary[cl] == args.T_names[s])]
            mean.append(mean_aux)
            raw_sum = np.sum(annotations_raw, axis=0)
            index = np.where(raw_sum == np.max(raw_sum))[0][0]

            # Plot activation over time
            colors = ['grey', 'sandybrown', 'orange', 'gold', 'olive', 'lightcoral', 'brown', 'red',
                        'limegreen', 'darkgreen', 'turquoise', 'darkcyan', 'blue', 'navy', 'purple', 'indigo', 'violet',
                        'deeppink', 'yellow', 'magenta', 'black']
            plt.subplots()
            for c in range(0, len(vocab)):
                plt.plot(annotations_raw[:,c], colors[c])
            plt.legend(vocab, fontsize=8)
            plt.title(vocab[index])
            plt.xlabel('Time', fontsize=15)
            plt.ylabel('Activation', fontsize=15)
            plt.tight_layout()
            #plt.savefig(args.output_dir + '/' + 'decoder_activation_' + str(trial_nr) + '_syll_' + vocab[index] + '.png')
            plt.close('all')

            # Scaling
            annotations_raw = annotations_raw[0:16,:]
            MAX = np.max(np.max(annotations_raw, axis=0)) # Find the max for each class, and take the max of the max
            index_MAX = np.where(np.max(annotations_raw, axis=0) == np.max(np.max(annotations_raw, axis=0))) # Where is the max in the annotations, in which syllable
            index_TIME = np.where(annotations_raw[:, index_MAX] == MAX)[0][0] # Where is the max in time, for a given syllables
            column_TIME = annotations_raw[index_TIME, :] # Remove values less than zero and bigger than one (piecewise function)
            for i in range(0, np.size(vocab)):
                if column_TIME[i] < 0.01:
                    column_TIME[i] = 0
                if column_TIME[i] > 1:
                    column_TIME[i] = 1
            max_expl_aux = np.zeros((np.size(args.T_names),))
            max_norm_expl_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                max_expl_aux[s] = column_TIME[np.where(vocab == args.T_names[s])]
                max_norm_expl_aux[s] = column_TIME[np.where(vocab == args.T_names[s])] / np.max(column_TIME)
            max_expl.append(max_expl_aux)
            max_norm_expl.append(max_norm_expl_aux)

            scaling_expl_aux = np.zeros((np.size(vocab),))
            for s in range(0, np.size(args.T_names)):
                scaling_expl_aux[s] = (column_TIME[s] - scaling[0][s][0]) / (scaling[0][s][1] - scaling[0][s][0])
            scaling_expl.append(scaling_expl_aux)

            p95_aux = np.zeros((np.size(vocab),))
            for s in range(0, np.size(args.T_names)):
                p95_aux[s] = column_TIME[s] / scaling[0][s][2]
            p95_expl.append(p95_aux)

            softmax_MAX_expl_aux = softmax_beta(5, column_TIME)
            sm = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                sm[s] = softmax_MAX_expl_aux[np.where(vocab == args.T_names[s])]
            softmax_MAX_expl.append(sm)

            scaling_softmax_expl_aux = softmax_beta(5, scaling_expl_aux)
            sm = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                sm[s] = scaling_softmax_expl_aux[np.where(vocab == args.T_names[s])]
            scaling_softmax_expl.append(sm)

            # Raw score
            raw_score_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                raw_score_aux[s] = (raw_sum[np.where(vocabulary[cl] == args.T_names[s])] - np.min(raw_sum)) / (
                            np.max(raw_sum) - np.min(raw_sum))
            raw_score_expl.append(raw_score_aux)

            # Standard logistic function
            sigm = 1/(1+np.exp(-raw_sum))
            sigmoid_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                sigmoid_aux[s] = sigm[np.where(vocabulary[cl] == args.T_names[s])]
            logistic_expl.append(sigmoid_aux)

            # Hyperbolic tangent
            tanh_aux = np.tanh(raw_sum)
            tanh_aux_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                tanh_aux_aux[s] = tanh_aux[np.where(vocabulary[cl] == args.T_names[s])]
            tanh_expl.append(tanh_aux_aux)

            # Sign normalization
            sign_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                sign_aux[s] = raw_sum[np.where(vocabulary[cl] == args.T_names[s])] / (1 + np.sign(raw_sum[np.where(vocabulary[cl] == args.T_names[s])])*raw_sum[np.where(vocabulary[cl] == args.T_names[s])])
            sign_expl.append(sign_aux)

            # Arctg normalization
            arctg_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                arctg_aux[s] = (2 / np.pi) * np.arctan((np.pi / 2) * raw_sum[np.where(vocabulary[cl] == args.T_names[s])])
            arctg_expl.append(arctg_aux)

            # Square normalization
            square_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                square_aux[s] = raw_sum[np.where(vocabulary[cl] == args.T_names[s])] / np.sqrt(1 + (raw_sum[np.where(vocabulary[cl] == args.T_names[s])])**2)
            square_expl.append(square_aux)

            # Mean normalization
            mean_norm_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                mean_norm_aux[s] = (raw_sum[np.where(vocabulary[cl] == args.T_names[s])] - np.mean(raw_sum)) / (
                        np.max(raw_sum) - np.min(raw_sum))
            mean_norm_expl.append(mean_norm_aux)

            # Minmax normalization
            minmax_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                minmax_aux[s] = raw_sum[np.where(vocabulary[cl] == args.T_names[s])] / (
                        np.max(raw_sum) - np.min(raw_sum))
            minmax_expl.append(minmax_aux)

            # Minmax sign normalization
            minmax_sign_aux = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                minmax_sign_aux[s] = np.sign(raw_sum[np.where(vocabulary[cl] == args.T_names[s])]) * raw_sum[np.where(vocabulary[cl] == args.T_names[s])] / (
                        np.max(raw_sum) - np.min(raw_sum))
            sign_minmax_expl.append(minmax_sign_aux)

            # Probability distribution with soft max (version with sum)
            A_all = softmax_beta(1, raw_sum)
            A = np.zeros((np.size(args.T_names),))
            for s in range(0, np.size(args.T_names)):
                A[s] = A_all[np.where(vocabulary[cl] == args.T_names[s])]
            softmax_sum_expl.append(A)

            # Probability distribution with soft max (version with mean)
            softmax_aux_collect = []
            for b in range(0, np.size(args.beta)):
                S_all = softmax_beta(args.beta[b], raw_mean)
                S = np.zeros((np.size(args.T_names),))
                for s in range(0, np.size(args.T_names)):
                    S[s] = S_all[np.where(vocabulary[cl] == args.T_names[s])]
                softmax_aux_collect.append(S)

            softmax_mean_expl.append(softmax_aux_collect)
            trial_nr = trial_nr + 1

        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_softmax_sum_expl_' + str(sim_counter) + '.npy', softmax_sum_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_softmax_mean_expl_' + str(sim_counter) + '.npy', softmax_mean_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_raw_score_expl_' + str(sim_counter) + '.npy', raw_score_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_mean_norm_expl_' + str(sim_counter) + '.npy', mean_norm_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_mean_expl_' + str(sim_counter) + '.npy', mean)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_logistic_expl_' + str(sim_counter) + '.npy', logistic_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_minmax_expl_' + str(sim_counter) + '.npy', minmax_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_sign_minmax_expl_' + str(sim_counter) + '.npy', sign_minmax_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_tanh_expl_' + str(sim_counter) + '.npy', tanh_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_sign_expl_' + str(sim_counter) + '.npy', sign_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_square_expl_' + str(sim_counter) + '.npy', square_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_arctg_expl_' + str(sim_counter) + '.npy', arctg_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_scaling_expl' + str(sim_counter) + '.npy', scaling_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_scaling_softmax_expl' + str(sim_counter) + '.npy', scaling_softmax_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_softmax_MAX_expl' + str(sim_counter) + '.npy', softmax_MAX_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_max_expl' + str(sim_counter) + '.npy', max_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_max_norm_expl' + str(sim_counter) + '.npy', max_norm_expl)
        np.save(args.output_dir + '/' + args.classifier_name[cl] + '_p95_expl' + str(sim_counter) + '.npy', p95_expl)

    print('Done')

def VLM_test(args):
    # Pre-define initial weights
    W_in = np.random.uniform(-args.W_seed, args.W_seed, (args.wavegan_latent_dim, np.size(args.T_names)))
    np.save(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy', W_in)

    # Pre-load exploration space
    all_motor = sorted(glob.glob(args.exploration_dir + '/' + '*.pkl'))

    # Pre-load perceptual space
    annotations_aux = []
    vocabulary = []
    scaling = []
    scaling_wo = []
    for cl in range(0, len(args.classifier_name)):
        annotations_aux.append(open_pkl(args.sensory_dir + '/' + 'sensory_' + str(args.wavegan_latent_dim) + '_' + args.classifier_name[cl] + '.pkl'))
        vocabulary.append(annotations_aux[cl][0].vocab)

        scaling_aux = np.load(args.sensory_dir + '/' + 'scaling_' + str(args.wavegan_latent_dim) + '_' + args.classifier_name[cl] + '.npy')
        scaling_wo_aux = np.zeros((np.size(args.T_names), 3))
        for s in range(0, np.size(args.T_names)):
            scaling_wo_aux[s, :] = scaling_aux[np.where(vocabulary[cl] == args.T_names[s]), :]
        scaling.append(scaling_aux)
        scaling_wo.append(scaling_wo_aux)

        # GAN classes (needed if classifier-EXT is used)
        if args.classifier_name[cl] != 'REAL':
            GAN_class = np.zeros((np.size(args.GAN_classes),))
            for gc in range(0, np.size(args.GAN_classes)):
                GAN_class[gc] = np.where(vocabulary[cl] == args.GAN_classes[gc])[0][0]

    for sim_counter in range(0,args.N_sim):
        # Create directory to save sensory productions
        os.makedirs(args.output_dir + '/' + args.sim_name + str(sim_counter))

        # Motor exploration: pick a random motor exploration
        expl = np.random.choice(range(np.size(all_motor)), args.MAX_trial, replace=True)
        np.save(args.output_dir + '/' + 'motor_exploration_sim_' + str(sim_counter) + '.npy', expl)

        for lr in range(0,np.size(args.learning_rate)):
            for cl in range(0, len(args.classifier_name)):
                # Initialization of the weights
                W_time_raw = []
                W_time_max = []
                W_time_max_norm = []
                W_time_max_scaling = []
                W_time_p95 = []

                W_raw = np.load(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy')
                W_soft = np.load(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy')
                W_max = np.load(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy')
                W_max_norm = np.load(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy')
                W_max_scaling = np.load(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy')
                W_p95 = np.load(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy')

                W_time_raw.append(W_raw)
                W_time_max.append(W_max)
                W_time_max_norm.append(W_max_norm)
                W_time_max_scaling.append(W_max_scaling)
                W_time_p95.append(W_p95)

                raw_score_time = []
                A_time = []
                max_time = []
                max_norm_time = []
                max_scaling_time = []
                p95_time = []
                A_expl = []
                if args.classifier_name[cl] != 'REAL':
                    A_GAN = []
                raw_score_expl = []
                max_score_expl = []
                max_norm_expl = []
                max_scaling_expl = []
                p95_expl = []
                A_expl_all =[]

                trial_nr = 0
                while trial_nr<args.MAX_trial:
                    M = open_pkl(all_motor[expl[trial_nr]])

                    # Sensory function (classifier)
                    annotations_raw = annotations_aux[cl][expl[trial_nr]].vect

                    # Raw score
                    raw_score_expl.append(auditory_activation('raw_score', annotations_raw, args.T_names, vocabulary[cl], scaling_wo[cl]))

                    # Probability distribution with soft max
                    A_all_aux = auditory_activation('softmax', annotations_raw, args.T_names, vocabulary[cl], scaling_wo[cl])
                    A_expl_all.append(A_all_aux)
                    A = np.zeros((np.size(args.T_names),))
                    for s in range(0,np.size(args.T_names)):
                        A[s] = A_all_aux[np.where(vocabulary[cl]==args.T_names[s])]
                    A_expl.append(A)
                    if args.classifier_name[cl] != 'REAL':
                        A_aux = []
                        for gc in range(0, np.size(args.GAN_classes)):
                            A_aux.append(A_all_aux[np.where(vocabulary[cl] == args.GAN_classes[gc])[0][0]])
                        A_GAN.append(A_aux)

                    # Max
                    aux = auditory_activation('max', annotations_raw, args.T_names, vocabulary[cl], scaling_wo[cl])
                    max_score_expl.append(aux[0])
                    max_norm_expl.append(aux[1])
                    max_scaling_expl.append(aux[2])
                    p95_expl_aux = np.zeros((np.size(args.T_names),))
                    for s in range(0,np.size(args.T_names)):
                        p95_expl_aux[s] = aux[0][s]/scaling_wo[cl][s][2]
                    p95_expl.append(p95_expl_aux)

                    # Define A
                    A_raw = raw_score_expl[trial_nr]
                    A_soft = A_expl[trial_nr]
                    A_max = max_score_expl[trial_nr]
                    A_max_norm = max_norm_expl[trial_nr]
                    A_max_scaling = max_scaling_expl[trial_nr]
                    A_p95 = p95_expl_aux

                    # Update the weights
                    W_raw, DeltaW = IM_simple_classic(args.learning_rate[lr], M[0], A_raw, W_raw, args.ns, args.wavegan_latent_dim)
                    W_soft, DeltaW = IM_simple_classic(args.learning_rate[lr], M[0], A_soft, W_soft, args.ns, args.wavegan_latent_dim)
                    W_max, DeltaW = IM_simple_classic(args.learning_rate[lr], M[0], A_max, W_max, args.ns, args.wavegan_latent_dim)
                    W_max_norm, DeltaW = IM_simple_classic(args.learning_rate[lr], M[0], A_max_norm, W_max_norm, args.ns, args.wavegan_latent_dim)
                    W_max_scaling, DeltaW = IM_simple_classic(args.learning_rate[lr], M[0], A_max_scaling, W_max_scaling, args.ns, args.wavegan_latent_dim)
                    W_p95, DeltaW = IM_simple_classic(args.learning_rate[lr], M[0], A_p95, W_p95, args.ns, args.wavegan_latent_dim)

                    W_all_aux = [W_raw, W_soft, W_max, W_max_norm, W_max_scaling, W_p95]

                    # Production of the sound using the WaveGAN as motor control function
                    if trial_nr % 15 == 0:
                        if not os.path.isdir(args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_' + args.sim_name + str(sim_counter) + '_' + str(trial_nr)):
                            os.makedirs(args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_' + args.sim_name + str(sim_counter) + '_' + str(trial_nr))
                        for i in range(0, args.ns):
                            if args.activation_motor == 'tanh':
                                motor_function_WaveGAN(args.train_dir + '/' + 'train_' + str(args.wavegan_latent_dim),
                                                       np.reshape(np.tanh(W[:, i]), (1, args.wavegan_latent_dim)),
                                                       '/' + 'sensory_production' + '_' + args.T_names[i] + '.wav',
                                                       args.sampling_rate,
                                                       args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                                       args.classifier_name[cl] + '_lr' + str(
                                                           args.learning_rate[lr]) + '_' + args.sim_name + str(
                                                           sim_counter) + '_' + str(trial_nr))

                            if args.activation_motor == 'piecewise':
                                for w in range(0, len(W_all_aux)):
                                    if not os.path.isdir(
                                            args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                            args.classifier_name[cl] + '_lr' + str(
                                                    args.learning_rate[lr]) + '_' + args.sim_name + str(
                                                    sim_counter)  + '_' + str(trial_nr) + '/' + '_' + '_condition_' + str(w) + '_' + str(trial_nr)):
                                        os.makedirs(args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                                    args.classifier_name[cl] + '_lr' + str(
                                            args.learning_rate[lr]) + '_' + args.sim_name + str(
                                            sim_counter) + '_' + str(trial_nr) + '/' + '_' + '_condition_' + str(w) + '_' + str(trial_nr))

                                    W = W_all_aux[w]
                                    for j in range(0, args.wavegan_latent_dim):
                                        if W[j, i] > 1:
                                            W[j, i] = 1
                                        elif W[j, i] < - 1:
                                            W[j, i] = - 1

                                    motor_function_WaveGAN(args.train_dir + '/' + 'train_' + str(args.wavegan_latent_dim), np.reshape(W[:, i], (1, args.wavegan_latent_dim)), '/' + 'sensory_production_condition_' + str(w) + '_' + args.T_names[i] + '.wav', args.sampling_rate, args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                                    args.classifier_name[cl] + '_lr' + str(
                                            args.learning_rate[lr]) + '_' + args.sim_name + str(
                                            sim_counter) + '_' + str(trial_nr) + '/' + '_' + '_condition_' + str(w) + '_' + str(trial_nr))

                        annotations_aux_gen = []
                        for w in range(0, len(W_all_aux)):
                            annotations_aux_gen_aux = sensory_response(args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                                        args.classifier_name[cl] + '_lr' + str(
                                                args.learning_rate[lr]) + '_' + args.sim_name + str(
                                                sim_counter) + '_' + str(trial_nr) + '/' + '_' + '_condition_' + str(w) + '_' + str(trial_nr))
                            annotations_aux_gen.append(annotations_aux_gen_aux)

                        A_gen = np.zeros((np.size(args.T_names),))
                        raw_score = np.zeros((np.size(args.T_names),))
                        max_score =  np.zeros((np.size(args.T_names),))
                        max_norm_score =  np.zeros((np.size(args.T_names),))
                        max_scaling_score =  np.zeros((np.size(args.T_names),))
                        p95_score =  np.zeros((np.size(args.T_names),))

                        for s in range(0, np.size(args.T_names)):
                            # Raw score
                            annotations_raw_gen = annotations_aux_gen[0][s].vect
                            vocab = annotations_aux_gen[0][0].vocab
                            raw_sum = np.sum(annotations_raw_gen, axis=0)
                            raw_score[s] = (raw_sum[np.where(vocab == args.T_names[s])]-np.min(raw_sum))/(np.max(raw_sum) - np.min(raw_sum))

                            # Softmax
                            annotations_raw_gen = annotations_aux_gen[1][s].vect
                            vocab = annotations_aux_gen[1][0].vocab
                            raw_sum = np.sum(annotations_raw_gen, axis=0)
                            A_gen[s] = scipy.special.softmax(raw_sum)[np.where(vocab == args.T_names[s])]

                            # Max
                            annotations_raw_gen = annotations_aux_gen[2][s].vect
                            vocab = annotations_aux_gen[2][0].vocab
                            aux = auditory_activation('max', annotations_raw_gen, args.T_names, vocab, scaling_wo[cl])
                            max_score[s] = aux[0][np.where(vocab == args.T_names[s])]

                            # Max norm
                            annotations_raw_gen = annotations_aux_gen[3][s].vect
                            aux = auditory_activation('max', annotations_raw_gen, args.T_names, vocab,
                                                      scaling_wo[cl])
                            max_norm_score[s] = aux[1][np.where(vocab == args.T_names[s])]

                            # Max scaling
                            annotations_raw_gen = annotations_aux_gen[4][s].vect
                            vocab = annotations_aux_gen[4][0].vocab
                            aux = auditory_activation('max', annotations_raw_gen, args.T_names, vocab,
                                                      scaling_wo[cl])
                            max_scaling_score[s] = aux[2][np.where(vocab == args.T_names[s])]

                            # p95
                            annotations_raw_gen = annotations_aux_gen[5][s].vect
                            vocab = annotations_aux_gen[5][0].vocab
                            aux = auditory_activation('max', annotations_raw_gen, args.T_names, vocab, scaling_wo[cl])
                            p95_score[s] = aux[0][np.where(vocab == args.T_names[s])]/scaling_wo[cl][s][2]

                        A_time.append(A_gen)
                        raw_score_time.append(raw_score)
                        max_time.append(max_score)
                        max_norm_time.append(max_norm_score)
                        max_scaling_time.append(max_scaling_score)
                        p95_time.append(p95_score)

                    W_time_raw.append(W_raw)
                    W_time_max.append(W_max)
                    W_time_max_norm.append(W_max_norm)
                    W_time_max_scaling.append(W_max_scaling)
                    W_time_p95.append(W_p95)

                    trial_nr = trial_nr + 1

                #print('Steps: ' + str(trial_nr - 1))

                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_W_raw_sim_' + str(sim_counter) + '.npy', W_time_raw)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_W_max_sim_' + str(sim_counter) + '.npy', W_time_max)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_W_max_norm_sim_' + str(sim_counter) + '.npy', W_time_max_norm)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_W_max_scaling_sim_' + str(sim_counter) + '.npy', W_time_max_scaling)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_W_p95_sim_' + str(sim_counter) + '.npy', W_time_p95)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_A_sim_' + str(sim_counter) + '.npy', A_time)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_raw_score_sim_' + str(sim_counter) + '.npy', raw_score_time)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_max_sim_' + str(sim_counter) + '.npy', max_time)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_max_norm_sim_' + str(sim_counter) + '.npy', max_norm_time)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_max_scaling_sim_' + str(sim_counter) + '.npy', max_scaling_time)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_p95_sim_' + str(sim_counter) + '.npy', p95_time)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_A_expl_' + str(sim_counter) + '.npy', A_expl)
                # np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_A_GAN_' + str(sim_counter) + '.npy', A_GAN)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_raw_score_expl_' + str(sim_counter) + '.npy', raw_score_expl)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_max_score_expl_' + str(sim_counter) + '.npy', max_score_expl)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_max_norm_expl_' + str(sim_counter) + '.npy', max_norm_expl)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_max_scaling_expl_' + str(sim_counter) + '.npy', max_scaling_expl)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_p95_expl_' + str(sim_counter) + '.npy', p95_expl)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_A_expl_all_' + str(sim_counter) + '.npy', A_expl_all)
                #np.save(args.output_dir + '/' + 'raw_eval_' + str(sim_counter) + '.npy', raw_evaluation)

    print('Done')

def VLM(args):
    # Pre-define initial weights (common across simulations)
    W_in = np.random.uniform(-args.W_seed, args.W_seed, (args.wavegan_latent_dim, np.size(args.T_names)))
    np.save(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy', W_in)

    # Pre-load exploration space
    all_motor = sorted(glob.glob(args.exploration_dir + '/' + 'motor_' + str(args.wavegan_latent_dim) + '/' + '*.pkl'))

    # Pre-load perceptual space
    annotations_aux = []
    vocabulary = []
    scaling = []
    scaling_wo = []
    for cl in range(0, len(args.classifier_name)):
        annotations_aux.append(open_pkl(args.sensory_dir + '/' + 'sensory_' + str(args.wavegan_latent_dim) + '_' + args.classifier_name[cl] + '.pkl'))
        vocabulary.append(annotations_aux[cl][0].vocab)

        scaling_aux = np.load(args.sensory_dir + '/' + 'scaling_' + str(args.wavegan_latent_dim) + '_' + args.classifier_name[cl] + '.npy')
        scaling_wo_aux = np.zeros((np.size(args.T_names), 3))
        for s in range(0, np.size(args.T_names)):
            scaling_wo_aux[s, :] = scaling_aux[np.where(vocabulary[cl] == args.T_names[s]), :]
        scaling.append(scaling_aux)
        scaling_wo.append(scaling_wo_aux)

        # GAN classes (needed if classifier-EXT is used)
        if args.classifier_name[cl] != 'REAL':
            GAN_class = np.zeros((np.size(args.GAN_classes),))
            for gc in range(0, np.size(args.GAN_classes)):
                GAN_class[gc] = np.where(vocabulary[cl] == args.GAN_classes[gc])[0][0]

    for sim_counter in range(0,args.N_sim):
        # Create directory to save sensory productions
        os.makedirs(args.output_dir + '/' + args.sim_name + str(sim_counter))

        # Motor exploration: pick a random motor exploration
        expl = np.random.choice(range(np.size(all_motor)), args.MAX_trial, replace=True)
        np.save(args.output_dir + '/' + 'motor_exploration_sim_' + str(sim_counter) + '.npy', expl)

        for lr in range(0,np.size(args.learning_rate)):
            for cl in range(0, len(args.classifier_name)):
                # Initialization of the weights
                W_time_p95 = []

                W_p95 = np.load(args.output_dir + '/' + 'W_in_' + str(args.wavegan_latent_dim) + '.npy')

                W_time_p95.append(W_p95)

                p95_time = []
                p95_expl = []

                trial_nr = 0
                while trial_nr<args.MAX_trial:
                    M = open_pkl(all_motor[expl[trial_nr]])

                    # Sensory function (classifier)
                    annotations_raw = annotations_aux[cl][expl[trial_nr]].vect

                    # Max
                    aux = auditory_activation('max', annotations_raw, args.T_names, vocabulary[cl], scaling_wo[cl])
                    p95_expl_aux = np.zeros((np.size(args.T_names),))
                    for s in range(0,np.size(args.T_names)):
                        p95_expl_aux[s] = aux[0][s]/scaling_wo[cl][s][2]
                        if p95_expl_aux[s] > 1:
                            p95_expl_aux[s] = 1
                    p95_expl.append(p95_expl_aux)

                    # Define A
                    A_p95 = p95_expl_aux

                    # Update the weights
                    W_p95, DeltaW = IM_simple_classic(args.learning_rate[lr], M[0], A_p95, W_p95, args.ns, args.wavegan_latent_dim)

                    W_all_aux = [W_p95]

                    # Production of the sound using the WaveGAN as motor control function
                    if trial_nr < 200 or trial_nr % 15 == 0:
                        if not os.path.isdir(args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_' + args.sim_name + str(sim_counter) + '_' + str(trial_nr)):
                            os.makedirs(args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_' + args.sim_name + str(sim_counter) + '_' + str(trial_nr))
                        for i in range(0, args.ns):
                            if args.activation_motor == 'tanh':
                                motor_function_WaveGAN(args.train_dir + '/' + 'train_' + str(args.wavegan_latent_dim),
                                                       np.reshape(np.tanh(W[:, i]), (1, args.wavegan_latent_dim)),
                                                       '/' + 'sensory_production' + '_' + args.T_names[i] + '.wav',
                                                       args.sampling_rate,
                                                       args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                                       args.classifier_name[cl] + '_lr' + str(
                                                           args.learning_rate[lr]) + '_' + args.sim_name + str(
                                                           sim_counter) + '_' + str(trial_nr))

                            if args.activation_motor == 'piecewise':
                                for w in range(0, len(W_all_aux)):
                                    if not os.path.isdir(
                                            args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                            args.classifier_name[cl] + '_lr' + str(
                                                    args.learning_rate[lr]) + '_' + args.sim_name + str(
                                                    sim_counter)  + '_' + str(trial_nr) + '/' + '_' + '_condition_' + str(w) + '_' + str(trial_nr)):
                                        os.makedirs(args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                                    args.classifier_name[cl] + '_lr' + str(
                                            args.learning_rate[lr]) + '_' + args.sim_name + str(
                                            sim_counter) + '_' + str(trial_nr) + '/' + '_' + '_condition_' + str(w) + '_' + str(trial_nr))

                                    W = W_all_aux[w]
                                    for j in range(0, args.wavegan_latent_dim):
                                        if W[j, i] > 1:
                                            W[j, i] = 1
                                        elif W[j, i] < - 1:
                                            W[j, i] = - 1

                                    motor_function_WaveGAN(args.train_dir + '/' + 'train_' + str(args.wavegan_latent_dim), np.reshape(W[:, i], (1, args.wavegan_latent_dim)), '/' + 'sensory_production_condition_' + str(w) + '_' + args.T_names[i] + '.wav', args.sampling_rate, args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                                    args.classifier_name[cl] + '_lr' + str(
                                            args.learning_rate[lr]) + '_' + args.sim_name + str(
                                            sim_counter) + '_' + str(trial_nr) + '/' + '_' + '_condition_' + str(w) + '_' + str(trial_nr))

                        annotations_aux_gen = []
                        for w in range(0, len(W_all_aux)):
                            annotations_aux_gen_aux = sensory_response(args.output_dir + '/' + args.sim_name + str(sim_counter) + '/' +
                                                        args.classifier_name[cl] + '_lr' + str(
                                                args.learning_rate[lr]) + '_' + args.sim_name + str(
                                                sim_counter) + '_' + str(trial_nr) + '/' + '_' + '_condition_' + str(w) + '_' + str(trial_nr))
                            annotations_aux_gen.append(annotations_aux_gen_aux)

                        p95_score =  np.zeros((np.size(args.T_names),))
                        for s in range(0, np.size(args.T_names)):
                            # p95
                            annotations_raw_gen = annotations_aux_gen[0][s].vect
                            vocab = annotations_aux_gen[0][0].vocab
                            aux = auditory_activation('max', annotations_raw_gen, args.T_names, vocab, scaling_wo[cl])
                            p95_score[s] = aux[0][np.where(vocab == args.T_names[s])]/scaling_wo[cl][s][2]
                            if p95_score[s] > 1:
                                p95_score[s] = 1

                        p95_time.append(p95_score)

                    W_time_p95.append(W_p95)

                    trial_nr = trial_nr + 1

                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_W_p95_sim_' + str(sim_counter) + '.npy', W_time_p95)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_p95_sim_' + str(sim_counter) + '.npy', p95_time)
                np.save(args.output_dir + '/' + args.classifier_name[cl] + '_lr' + str(args.learning_rate[lr]) + '_p95_expl_' + str(sim_counter) + '.npy', p95_expl)

    print('Done')

if __name__ == '__main__':
    """
    Example how to run it:
    > python InverseModelGAN.py --option learning --output_dir OUTPUT_DIR --wavegan_latent_dim 3 --ckpt_n CKPT --MAX_trial 3001
    """

    import argparse
    import glob
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str,
                        choices=['target', 'learning', 'seed', 'scaling', 'auditory_act'])
    parser.add_argument('--exploration_dir', type=str,
                        help='Motor exploration directory: directory containing the latent vectors and'
                             'the corresponding generated wav files',
                        default='motor')
    parser.add_argument('--sensory_dir', type=str,
                        help='Sensory response feedback directory containing annotations and eventually other related analysis',
                        default='sensory')
    parser.add_argument('--template_dir', type=str,
                        help='Real syllable name: objective of the learning without extension',
                        default='templates')
    parser.add_argument('--train_dir', type=str, help='Train directory: where the trained model (GAN) is saved',
                        default='train')
    parser.add_argument('--output_dir', type=str, help='Output directory: to store wav files and other useful variables',
                        default=None)
    parser.add_argument('--seed_dir', type=str, help='Directory containing the pre-computed initial values - to use a common seed', default='seed/ALLseed0001') #'Seed5syll') #'ALLseed0001')  #'Seed_3syll\seed0001')

    wavegan_args = parser.add_argument_group('WaveGAN')
    wavegan_args.add_argument('--wavegan_latent_dim', type=int,
                              help='Dimension of the latent space',
                              default = 2)
    wavegan_args.add_argument('--sampling_rate', type=int,
                              help='Sampling rate: the same used during the training of the GAN',
                              default=16000)
    wavegan_args.add_argument('--ckpt_n', type=int,
                              help='At which chekpoint it has to be saved. And the first line in the checkpoint file has to be changed for model_ckpt=ckpt_n',
                              default=False)

    IM_args =parser.add_argument_group('IMlearning')
    IM_args.add_argument('--learning_rate', type=list,
                         help='Learning rate used during learning',
                         default = [0.1, 0.01]) # [0.1, 0.01]

    simulation_args = parser.add_argument_group('Simulation')
    simulation_args.add_argument('--MAX_trial', type=int,
                                 help='Maximal number of trials',
                                 default = 3001)
    simulation_args.add_argument('--ns', type=int,
                                 help='number of syllables',
                                 default = 16)
    simulation_args.add_argument('--W_min', type=float,
                                 help='lower boundary of the latent space',
                                 default = -1)
    simulation_args.add_argument('--W_max', type=float,
                                 help='lower boundary of the latent space',
                                 default = 1)
    simulation_args.add_argument('--W_option', type=str, help='Which learning rule: if simple (default) is traditional Hebbian learning rule, whereas '
                                                              'norm stands for descreasing factor normalized learning rule', default='simple')
    simulation_args.add_argument('--W_seed', type=float, help='Weight inizialization', default=0.001)
    simulation_args.add_argument('--N_sim', type=int, help='Number of instances', default=3)
    simulation_args.add_argument('--T_names', type=list, help='Target syllables', default=['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']) #['A', 'B1', 'B2', 'C', 'D', 'E', 'H', 'J1', 'J2', 'L', 'M', 'N', 'O', 'Q', 'R', 'V']) # default=['B1', 'C', 'M'])
    simulation_args.add_argument('--sim_name', type=str, help='Sub directory containing the generations per each simulation', default='sensory_prod_sim_')
    simulation_args.add_argument('--classifier_name', type=list, help='Which classifier model I want to use. Multiple classifier are allowed', default=['EXT']) #'EXT', 'REAL'
    simulation_args.add_argument('--GAN_classes', type=list, help='GAN classes in the decoder', default=['EARLY15', 'EARLY30', 'EARLY45', 'OT', 'WN'])
    simulation_args.add_argument('--activation_motor', type=str, help='Type of motor activation', choices=['linear', 'piecewise', 'tanh'], default='piecewise')
    simulation_args.add_argument('--beta', type=list, help='Type of auditory softmax activation', default=[1, 5])

    plot_args = parser.add_argument_group('Plots')
    plot_args.add_argument('--format', type=str, help='Saving format', default='png')

    args = parser.parse_args()

    # Make output dir
    if args.output_dir != None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        # Save args
        with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
            f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    if args.option == 'target':
        target(args)

    if args.option == 'learning':
        VLM(args)

    if args.option == 'seed':
        pre_def(args)

    if args.option == 'scaling':
        exploration_space(args)

    if args.option == 'auditory_act':
        auditory_activation_test(args)
