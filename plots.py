import numpy as np
import scipy as sp
import scipy.io.wavfile as wav
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_distribution_classes(X, my_ticks, title, n_classes):
    """
    :param X: data
    :param my_ticks: personal ticks for x label (name of the classes)
    :param title: title of the plot
    :param n_classes: number of classes

    :return: the figure, the distribution and its features (mean, standard deviation, variance)
    """
    h, bins = np.histogram(X, bins=range(n_classes+1))
    h_mean = np.mean(h)
    h_std = np.std(h)
    h_var = np.var(h)
    fig, ax = plt.subplots()
    ax.bar(bins[:-1], h, width=0.8, color='b', alpha=0.6, label='Syllable classes distribution', align='center')
    plt.plot(bins[:-1], np.ones((n_classes))*h_mean, 'k')
    ax.fill_between(bins[:-1], h_mean - h_std, h_mean + h_std, 'k', alpha=0.2)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.legend(loc='upper right', fontsize=8, ncol=1, shadow=True, fancybox=True)
    plt.xticks(bins[:-1], my_ticks, rotation='vertical', fontsize=6)
    plt.xlabel('Classes of syllables')
    plt.ylabel('Number of occurences')
    #plt.title(title)

    return fig, h, h_mean, h_std, h_var

def plot_correlation_general(correlation_matrix, x_label, y_label, title):
    """
    :param correlation_matrix: pre-computed correlation matrix
    :param template_name: the name of the classes
    :param title: title of the plot

    :return: the cross correlation matrix figure
    """
    fig, ax = plt.subplots()
    plt.imshow(correlation_matrix, cmap=plt.cm.Blues, aspect='auto')
    plt.xticks(range(np.size(x_label)), x_label, fontsize=7)
    plt.yticks(range(np.size(y_label)), y_label, fontsize=7)
    plt.colorbar()
    #plt.clim(0, 1)
    plt.tight_layout()  # to avoid the cut of labels
    plt.title(title)

    return fig


def plot_histogram(X, n_bins, my_color, title):
    """
    :param X: data
    :param n_bins: number of bins
    :param my_color: my_color
    :param title: title of the plot

    :return: histogram of the data
    """
    fig, ax = plt.subplots()
    plt.hist(X, n_bins, color= my_color, alpha=0.6)
    plt.title(title)

    return fig

def plot_spectro(filename, window, nperseg, title):
    """
    :param filename: name of the file to plot (.wav file)
    :param window: as in the documentation of scipy.signal.spectrogram
    :param nperseg: scipy.signal.spectrogram
    :param title: title of the figure

    :return: the spectrogram figure
    """
    sr, samples = wav.read(filename)

    fig, ax = plt.subplots()
    (f, t, spect) = sp.signal.spectrogram(samples, sr, window, nperseg, nperseg - 64, mode='complex')
    ax.imshow(10 * np.log10(np.square(abs(spect))), origin="lower", aspect="auto", interpolation="none", extent=[0, max(t) * 200, min(f), max(f)], cmap='inferno')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (ms)')
    plt.title(title)

    return fig

