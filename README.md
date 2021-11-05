Link to the paper:

# What to do first
* Train the GAN using ld = latent space dimension (e.g., ld = 3)
* Generate the motor space (here 16k generations): name it 'motor_ld' (e.g., 'motor_3')
* Create annotations (analysis of the motor space): name it 'sensory_EXT_ld.pkl' (e.g., 'sensory_EXT_3.pkl')


* [OPTIONAL]: * use `pre_def` to pre-define the initial weights. This is useful if one can't run for a long time the simulations.

---
# Training
## Prepare main directories 
* _exploration_dir_: Motor exploration directory: directory containing the latent vectors and the corresponding generated wav files.
* _sensory_dir_: Sensory response feedback directory containing annotations (sensory_EXT_3.pkl) and eventually other related analysis.
* _train_dir_: Train directory: where the trained model (GAN) is saved.

---
## Structural functions
### Learning rule
`IM_simple_classic`: implementation of classic Hebbian learning rule

### Motor control function
`motor_function_WaveGAN`: generator of WaveGAN, only generative part since the training is done a priori.

This function generates the sound starting from a vector having the same dimension of the latent space: it has to be fixed before to start the WaveGAN training.
To use this function the train directory of the model needs to be available in order to get access to the
checkpoint.

It returns one vocal generation: for example, if I want to associate on vector to one syllable than this
function takes as input one vector (saved as pkl) and generates one syllable.

Partially taken from train_wavegan.py in https://github.com/spagliarini/low-dimensional-canary-GAN (original WaveGAN is from https://github.com/chrisdonahue/wavegan).

### Sensory response function
`sensory_response' is classifier-EXT (create annotations).

The input directory contains one or more audio files (.wav) of duration 1s.

This function creates a dictionary containing three elements:
    - mean: this entry strores the averaged outputs produced by the ESN, for each sample. This gives each sample a unique annotation vector;
            the mean output link the whole output (the whole syllable) to a 16 components vector (which is an indicator vector for the 16 classes of syllables)
    - raw: this entry stores the raw outputs produced by the ESN. The ESN produce one annotation vector per timestep;
           the raw output does the same thing but for each timestep of input audio
    - states: this entry stores the internal states of the ESN over time.

The output is saved in the same directory of the data.

References for the classifier:
- link to git project: https://github.com/reservoirpy/reservoirpy
- link to ICANN paper: https://github.com/neuronalX/Trouvain2020_ICANN

### Auditory activation
`auditory_activation`: to compute the second layer of the sensory response function.
Possibilities:
- softmax
- max scaling
- p95 (we use thin one)

### Main
`VLM`: learning model.

Main parameters (at the end of the code):
- _wavegan_latent_dim_: latent space dimension
- _sampling_rate_: to write the syllables (same of training/exploration data)
- _ckpt_n_: At which chekpoint it has to be saved. And the first line in the checkpoint file has to be changed for model_ckpt=ckpt_n.
- _learning_rate_
- _MAX_trial_: max number of trials per simulation
- _ns_: number of syllable to learn
- _W_min_: min boundary for the weights (used int the motor activation which is piecewise linear)
- _W_max_: max boundary for the weights (used int the motor activation which is piecewise linear)
- _W_option_: which learning rule 
- _W_seed_: weights initialization
- _N_sim_: number of simulations to run in a row (with the same initial weights)
- _T_names_: names of the syllables 
- _classifier_name_: which classifier (now we use only EXT)
- _activation_motor_: we now use piecewise.
                      
    python InverseModelGAN.py --option learning --output_dir OUTPUT_DIR --wavegan_latent_dim 3 --ckpt_n CKPT --MAX_trial 3001

---
## Utils
* `open_pkl` : to open annotations (sensory response) files.
* `softmax_beta`: to compute the softmax varying the parameter beta.
* `auditory_activation_test`: to explore different types of auditory activation on the motor space.
* `exploration_space`: representation of 1D and 3D motor space (cube, slices, etc.)
* `VLM_test`: to test the learning with ALL the auditory activation function. Ok for a limited number of activation functions and iterations.
* `target`: Function to select the target (ideally the syllables that activate the most the classifier).

---
# Figures 
Function `plotGAN` contains several function to plot the results of the learning model.

Common important parameters:
* _data_dir_: where the data are
* _output_dir_: where to save the figures
* _wavegan_latent_dim_: latent space dimension
* _MAX_trial_: Max number of time steps (the same as the one used during training)
* _ns_: number of syllable to learn
* _N_sim_: number of simulations to run in a row (with the same initial weights)
* _classifier_name_: which classifier (now we use only EXT)
* _learning_rate_: list of the learning rates used during training
* _T_names_: names of the syllables 

`plot_auditory_activation`: Plot the results of the different auditory activation functions (results from the test function).
Additional parameters:
* _beta_: parameter for the softmax function


    python plotGAN.py --option activation_aud --data_dir DATA_DIR --outuput_dir OUTPUT_DIR --MAX_trial 3001


`plot_sensory`: Plots of the results obtained from the leanring model (VLM function in `InverseModelGAN`).
Additional parameters:
* _n_points_: How many saved points (every a certain number of epochs. E.g., 300).


    python plotGAN.py --option sensory --data_dir DATA_DIR --outuput_dir OUTPUT_DIR --MAX_trial 3001 --n_points 300

`plot_syll`: Plot the example of a syllable at a certain time steps (example spectrograms): change the name in syllables variable (just at the beginning of the function).


    python plotGAN.py --option syll --data_dir DATA_DIR --outuput_dir OUTPUT_DIR 

`mean_spectro`: compute and plot the mean spectrogram during plateau.
Additional parameters:
* _N_: Nftt spectrogram librosa
* _H_: Hop length spectrogram librosa
* _color_: colormap

Note: change the parameters (bottom of the code) to plot the correct one (it could take some time so it is better to do it one by one).

    python plotGAN.py --option mean_spectro --data_dir DATA_DIR --outuput_dir OUTPUT_DIR --n_points 300
    
`cfr_dim13`: comparison between different latent space size. To be updated when more are available. Might be general for comparisons (e.g., also between sparse/not sparse).
This function saves in the input directory.

    python plotGAN.py --option cfr --data_dir DATA_DIR --MAX_trial 3001
    
`plot_sensory_test`: Plots of the results obtained from the learning model (VLMtest function in `InverseModelGAN`). This is not good for long simulations (it takes a lot of time), it's ok to run short experiments.



