## LIBRARIES
from autograd import grad
import autograd.numpy as np
np.set_printoptions(suppress=True)
import imageio 
import matplotlib.pyplot as plt
import os
from psychopy import core, event, gui, monitors, visual
import pandas as pd
from sklearn.decomposition import PCA
import six
import string
# import tkinter as tk

## FUNCTIONS

pca = PCA(n_components=2)

def prompt(dataframe):

    functions = [option for option in dataframe['fx'] if str(option) != 'nan']
    problems = [option for option in dataframe['problem']]

    myDlg = gui.Dlg(title = 'Run parameters')
    myDlg.addField('Activation fx:', choices=functions)
    myDlg.addField('Problem:', choices=problems)
    myDlg.addField('Hidden L-rate:', dataframe['lr1'][0])
    myDlg.addField('Output L-rate:', dataframe['lr2'][0])
    myDlg.addField('Initial wt range:', dataframe['in_wtr'][0])
    myDlg.addField('Hidden nodes:', dataframe['n_hids'][0])
    myDlg.addField('Epochs:', dataframe['n_epochs'][0])
    myDlg.addField('Iterations:', dataframe['n_itr'][0])
    myDlg.addField('Hidden layer max:', dataframe['max'][0])
    myDlg.addField('sensitivity:', dataframe['c'][0])
    myDlg.addField('Response mapping:', dataframe['map'][0])
    myDlg.show()

    # quit if participant does not click 'ok' or pnum blank
    if not myDlg.OK:
        core.quit()
    else:
        return myDlg, functions

def preprocessing(problem):

    path = os.path.join('datasets','{}.csv'.format(problem))
    data = np.genfromtxt(path, delimiter=',')

    inputs = data[:,:-1]
    labels = data[:,-1]

    n_classes = len(np.unique(labels))
    n_dims = inputs.shape[1]

    # one-hot code targets
    if np.min(labels) != 0:
        labels -= 1 # need dummy code to start at zero for this to work
    labels = labels.astype(int)
    labels = np.eye(n_classes)[np.array(labels)]

    # norm data to be between -1 and 1
    if problem[:-1] != 'shj':
        inputs -= np.min(inputs, axis = 0)
        inputs /= np.ptp(inputs, axis = 0)
        inputs *= 2
        inputs -= 1
        full_set = np.append(inputs, labels, 1)
    else:
        full_set = np.append(inputs, labels, 1)
        full_set = np.concatenate((full_set, full_set), axis=0) # to match Nosofsky+ '94

    return [full_set, n_classes, n_dims]

# def prepLoops(problem):

#     path = os.path.join('datasets','{}.csv'.format(problem))
#     data = np.genfromtxt(path, delimiter=',')

#     inputs = data[:,:-1] # drop test data
#     targets = data[:,-1] - 1
    
#     n_class = len(np.unique(targets))
#     n_dims = inputs.shape[1]

#     # SCALE DATA
#     inputs = np.round((inputs / inputs.max(axis=0)), 3)

#     # one hot coding
#     targets = targets.astype(int)
#     targets = np.eye(n_class)[np.array(targets)]

#     # RECOMBINE
#     stitch = np.append(inputs, targets, 1)

#     # ONLY TRAIN
#     train = stitch[:20,:]

#     # # ONLY TEST
#     # test = stitch[20:,:]

#     # problem = [train, test]

#     return [train, n_class, n_dims]

def gen_weights(n_dims, n_classes, parameters):

    in_weights = np.random.uniform(low=-parameters['in_wtr'][0], high=parameters['in_wtr'][0], size=(n_dims, parameters['n_hids'][0]))
    out_weights = np.random.uniform(low=-0.1, high=0.1, size=(parameters['n_hids'][0], n_classes))

    return [in_weights, out_weights]

def gen_weights_cl(n_dims, n_classes, parameters):

    in_weights = np.random.uniform(low=-parameters['in_wtr'][0], high=parameters['in_wtr'][0], size=[n_dims, 1, parameters['n_hids'][0]])
    out_weights = np.random.uniform(low=-0.1, high=0.1, size=(parameters['n_hids'][0], n_classes))

    return [in_weights, out_weights]

def gen_weights_ch(n_dims, n_classes, parameters):

    in_weights = np.full([n_dims, parameters['n_hids'][0]], parameters['c'][0])
    in_bias = np.random.uniform(-1, 1, size=[parameters['n_hids'][0], 1, n_dims])
    out_weights = np.random.normal(-0.1, 0.1, size=(parameters['n_hids'][0], n_classes))

    return [in_bias, out_weights, in_weights]

def update_weights(wts, gradients, parameters):

    wts[0] -= parameters['lr1'][0] * gradients[0]
    wts[1] -= parameters['lr2'][0] * gradients[1]

    wts[0][wts[0] > parameters['max'][0]] = parameters['max'][0]
    wts[0][wts[0] < -parameters['max'][0]] = -parameters['max'][0]
    wts[1][wts[1] > 1e+100] = 1e+100
    wts[1][wts[1] < -1e+100] = -1e+100

    return wts

def update_weights_ch(wts, gradients, parameters):

    wts[0] -= parameters['lr1'][0] * gradients[0]
    wts[1] -= parameters['lr2'][0] * gradients[1]
    wts[2] -= parameters['lr1'][0] * gradients[2]

    wts[0][wts[0] > parameters['max'][0]] = parameters['max'][0]
    wts[0][wts[0] < -parameters['max'][0]] = -parameters['max'][0]
    wts[1][wts[1] > 1e+100] = 1e+100
    wts[1][wts[1] < -1e+100] = -1e+100

    return wts
    
def softmax(hidden_activations, wts, labels, parameters):

    index = np.argmax(labels).astype(int)

    # output layer inputs
    output_dotproduct = np.dot(hidden_activations[0], wts[1])
    mapped = output_dotproduct*parameters['map'][0]
    
    e = np.exp(output_dotproduct - output_dotproduct.max())
    e_map = np.exp(mapped - mapped.max())

    probabilities = []
    
    for data in [e, e_map]:
        
        probabilities.append((data / data.sum()))

    return [probabilities[0][index], probabilities[1][index], probabilities[1]] 

def warp_classic(wts, inputs, labels, parameters):

    # hidden layer input
    minus_inputs = wts[0].T - inputs
    dot_product = np.einsum('hif, hif -> ih', minus_inputs, -minus_inputs)

    # hidden layer activations
    hidden_activations = np.exp(parameters['c'][0] * dot_product)

    # output layer activations
    output_activations = softmax(hidden_activations, wts, labels, parameters)

    return output_activations, hidden_activations

def cherry_warp(wts, inputs, labels, parameters):

    # hidden layer input
    biased_inputs = np.subtract(inputs, wts[0])**2

    # hidden layer activations  
    hidden_activations = np.exp(-np.absolute(np.einsum('hif,fh -> ih', biased_inputs, wts[2])))
    
    # output layer activations
    output_activations = softmax(hidden_activations, wts, labels, parameters)

    return output_activations, hidden_activations

def vanilla_warp(wts, inputs, labels, parameters):

    # hidden layer input
    negative_SSD = -np.sum((inputs - wts[0].T)**2, 1)

    # hidden layer activation
    hidden_activations = np.exp(parameters['c'][0] * negative_SSD)

    # output layer activations
    output_activations = softmax(hidden_activations[np.newaxis,:], wts, labels, parameters)

    return output_activations, hidden_activations

def loss(wts, inputs, labels, parameters, act_fx):

    output_activations, hidden_activations = act_fx(wts, inputs, labels, parameters)
    cross_entropy = -np.log(output_activations[0])

    return cross_entropy

optimizer = grad(loss)


def run_model(problem, parameters, full_set, n_classes, n_dims, wt_gen, wt_update, act_fx):

    # init matrix to store run performance
    run_performance = np.zeros((parameters['n_itr'][0], parameters['n_epochs'][0]))

    # begin iterating through n_iterations
    for iteration in range(parameters['n_itr'][0]):

        # create weights, init list for current iteration performance
        wts = wt_gen(n_dims, n_classes, parameters)
        iteration_performance = []

        # begin iterating through n_epochs
        for epoch in range(parameters['n_epochs'][0]):

            # shuffle stim set for current epoch, init list for item performance
            np.random.shuffle(full_set)
            item_performance = []


            # begin iteration through n_items
            for row in full_set:

                item = np.array(row[:-n_classes][np.newaxis,:])
                label = np.array(row[-n_classes:][np.newaxis,:])

                # return performance, hidden activations
                output_activations, hidden_activations = act_fx(wts, item, label, parameters)
                item_performance.append(1 - output_activations[1])

                # update weights
                gradients = optimizer(wts, item, label, parameters, act_fx)
                wts = wt_update(wts, gradients, parameters)

            # append average of epoch
            iteration_performance.append(np.mean(item_performance))

        # append average of iteration
        run_performance[iteration,:] = iteration_performance

    # determine average performance for each block across iterations
    run_performance = np.mean(run_performance, axis=0)
    run_metrics = [problem, parameters['n_itr'][0], parameters['n_epochs'][0], np.round(run_performance, 3)]
    performance_statement = ('\n' + 'Run completed (activation function: {}, problem: {})\n\nWith {} iteration(s) and {} epoch(s),'
                            ' average error over iterations was:\n\n{}\n\nPress SPACE to continue, or press ESCAPE to exit'.format(parameters['fx'][0], *run_metrics))

    return performance_statement, output_activations, wts

def draw_and_wait(win, text_var, string):

    text_var.setText(string)
    text_var.draw()
    win.flip()

    resume=event.waitKeys(keyList=['space', 'escape'])
    if resume[0]=='escape':
        core.quit()

def get_item(win, text, text_2, n_dims, key_names):

    container = []
    response = ''
    instructions = ('Use the keyboard to enter a set of input values to test. Keep the values comma separated with no spaces. '
                    'Make sure to include {} feature values. Press RETURN to advance, ESCAPE to exit, or type "gif" then RETURN to create a gif (may take a while).'.format(str(n_dims)))


    text.setText(instructions)
    text.setPos([0, 0])
    text_2.setText(response)
    text_2.setPos([0, -100])
    text.draw()
    text_2.draw()
    win.flip()

    ## LOOP TO COLLECT RESPONSE
    while True:

        key_press = event.waitKeys()
        
        # quit if user escapes
        if key_press == ['escape']:
            win.close()
            core.quit()
        elif key_press == ['return']:
            if len(response) > 0:
                break
            else:
                pass
        elif key_press == ['backspace']: # delete last item if user pressed backspace
            if len(container) == 0: # but don't do anything if they haven't typed anything in yet
                pass
            else:
                container = container[:-1]
        elif key_press[0] in key_names.keys():
            container.append(key_names[key_press[0]])
        elif key_press[0] in string.digits + 'gif':
            container.append(key_press[0])

        response = ''.join(container)
        text_2.setText(response)
        text.draw()
        text_2.draw()
        win.flip()

    return response 

# with credit to volodymyr on StackOverflow for providing the majority of the following function code
def render_mpl_table(data, fx, problem, col_width=2.5, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    # define paths and save
    output_dir = os.path.join(os.getcwd(), 'visuals')
    png = os.path.join(output_dir, '{}_{}.png'.format(fx, problem))
    figure_file = ax.get_figure()
    figure_file.savefig(png)

    return png


def run_query(win, inputs, wts, parameters, n_classes, n_dims, act_fx, text, text_2, img):

    # for convenience
    n_hids = parameters['n_hids'][0]

    # retrieve outputs
    item = np.array([float(integer) for integer in inputs.split(',')])
    output_activations, hidden_activations = act_fx(wts, item, np.array([0.]), parameters)
    softmax_dist = list(np.round(output_activations[2], 3))

    # start assembling text message
    set_up = 'With {} as input, the softmax distribution was '.format(inputs)

    dist_list = []
    for n in range(n_classes):
        dist_list.append('Category {}: {}'.format(string.ascii_uppercase[n], softmax_dist[n]))
    dist_string = ', '.join(dist_list)

    column_names = []
    for n in range(n_dims):
        column_names.append('F{}'.format(n+1))
    column_names.append('activations')

    for n in range(n_classes):
        column_names.append('category {}'.format(string.ascii_uppercase[n]))

    ow_associations = np.exp(wts[1] - wts[1].max()) / np.sum(np.exp(wts[1] - wts[1].max()), 1).reshape(n_hids, 1)

    wts_activation = np.c_[(wts[0].reshape(n_hids, n_dims), hidden_activations.reshape(n_hids, 1))]
    wts_classes = np.c_[wts_activation, ow_associations]

    wts_dataframe = pd.DataFrame(np.round(wts_classes, 3), columns=column_names, index=range(1, n_hids+1))
    wts_dataframe.insert(0, 'node', wts_dataframe.index)
    wts_dataframe = wts_dataframe.sort_values('activations', ascending=False)
    wts_dataframe = wts_dataframe[0:min(10, n_hids)]

    filename = render_mpl_table(wts_dataframe, parameters['fx'][0], parameters['problem'][0])

    wts_statement = '\n\nWeights and activations (top 10) were:\n\n'

    run_info = set_up + dist_string + wts_statement

    advance = '\n\n Press SPACE to return to the last screen.'

    text.setText(run_info)
    text.setPos([0, 350])
    text_2.setText(advance)
    text_2.setPos([0, -350])
    img.setImage(filename)

    img.draw()
    text.draw()
    text_2.draw()

    win.winHandle.activate() # needed because matplotlib steals focus
    win.flip()
    
    resume=event.waitKeys(keyList=['space', 'escape'])
    if resume[0]=='escape':
        core.quit()


def wt_watcher(parameters, full_set, n_classes, n_dims, wt_gen, wt_update, act_fx):

    # create weights, empty lists for later
    wts = wt_gen(n_dims, n_classes, parameters)
    iteration_performance = []
    activations = []
    wt_history = np.zeros((parameters['n_epochs'][0], parameters['n_hids'][0], n_dims+n_classes+1))

    # begin iterating through n_epochs
    for epoch in range(parameters['n_epochs'][0]):

        # shuffle stim set for current epoch, init list for item performance
        np.random.shuffle(full_set)
        item_performance = []

        # begin iteration through n_items
        for row in full_set:

            item = np.array(row[:-n_classes][np.newaxis,:])
            label = np.array(row[-n_classes:][np.newaxis,:])

            # return performance, hidden activations
            output_activations, hidden_activations = act_fx(wts, item, label, parameters)
            item_performance.append(1 - output_activations[1])
            activations.append(np.round(hidden_activations, 2))
            
            # update weights
            gradients = optimizer(wts, item, label, parameters, act_fx)
            wts = wt_update(wts, gradients, parameters)
            
        # append average of epoch
        iteration_performance.append(np.mean(item_performance))

        ## CREATE ARP MATRIX
        if parameters['fx'][0] == 'finite mixutre':
            arp_matrix = wts[0].reshape(parameters['n_hids'][0], n_dims)
        else:
            arp_matrix = wts[0].T

        activations_matrix = np.vstack(activations)
        activations_max = activations_matrix.max(0)
        resp_probs = np.exp(wts[1] - wts[1].max()) / np.sum(np.exp(wts[1] - wts[1].max()), 1).reshape(parameters['n_hids'][0], 1)

        arp_matrix = np.c_[arp_matrix, activations_max, resp_probs]
        
        wt_history[epoch,:,:] = arp_matrix

    full_copy = np.copy(full_set, order='K')
    full_copy = np.insert(full_copy, -n_classes, values=np.ones(full_copy.shape[0]), axis=1)

    ## COMPRESS DATASET IF N_DIMS GREATER THAN 3
    if n_dims > 3:

        inputs_copy = full_copy[:,:n_dims]
        inputs_copy = (inputs_copy - inputs_copy.min())/(inputs_copy.max() - inputs_copy.min())

        all_hiddens = wt_history.reshape(parameters['n_epochs'][0] * parameters['n_hids'][0], n_dims+n_classes+1)
        all_hiddens_trunc = all_hiddens[:,:n_dims]
        all_hiddens_trunc = (all_hiddens_trunc - all_hiddens_trunc.min())/(all_hiddens_trunc.max() - all_hiddens_trunc.min())
        
        pca_full_set = pca.fit_transform(np.r_[inputs_copy, all_hiddens_trunc])

        pca_copy = pca_full_set[:inputs_copy.shape[0],:]
        full_copy = np.c_[pca_copy, full_copy[:,n_dims:]]

        arp_matrix = pca_full_set[inputs_copy.shape[0]:,:]
        arp_matrix = np.c_[arp_matrix, all_hiddens[:,n_dims:]]

        wt_history = arp_matrix.reshape(parameters['n_epochs'][0], parameters['n_hids'][0], n_classes+3)

    new_dims = arp_matrix.shape[1] - (n_classes + 1)

    return wt_history, full_copy, iteration_performance, new_dims
    
def two_ft_gif(win, full_copy, problem, n_hids, wt_history, iteration_performance, n_classes):


    if n_classes == 2:

        for epoch in range(wt_history.shape[0]):


            # collect plot min/max
            combined_df = np.r_[full_copy[:,:2], wt_history[epoch,:,:2]]

            x_min = combined_df[:,0].min()
            x_max = combined_df[:,0].max()
            y_min = combined_df[:,1].min()
            y_max = combined_df[:,1].max()

            # remove extraneous categories, leaving only prob(A)
            snapshot_df = pd.DataFrame(wt_history[epoch,:,:-1])
            original_data = pd.DataFrame(full_copy)

            # create scatterplot
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(snapshot_df.loc[:,0], snapshot_df.loc[:,1], s=snapshot_df.loc[:,2]*200, c=snapshot_df.loc[:,3], cmap='RdBu', marker='o')
            ax.scatter(original_data.loc[:,0], original_data.loc[:,1], s=original_data.loc[:,2]*100, c=original_data.loc[:,3], cmap='RdBu', marker='P')
            ax.set_title('{}, epoch{}, error:{}'.format(problem, epoch+1, np.round(iteration_performance[epoch], 2)))

            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])

            # define paths and save
            output_dir = os.path.join(os.getcwd(), 'visuals')
            png = os.path.join(output_dir, 'gif_{}_{}.png'.format(problem, epoch))
            figure_file = ax.get_figure()
            figure_file.savefig(png)

    else:

        for epoch in range(wt_history.shape[0]):

            # collect plot min/max
            combined_df = np.r_[full_copy[:,:2], wt_history[epoch,:,:2]]

            x_min = combined_df[:,0].min()
            x_max = combined_df[:,0].max()
            y_min = combined_df[:,1].min()
            y_max = combined_df[:,1].max()

            # remove extraneous categories, leaving only prob(A)
            snapshot_df = pd.DataFrame(wt_history[epoch,:,:])
            original_data = pd.DataFrame(full_copy)

            # reverse the one-hot coding to get integer categories
            snapshot_df['category'] = snapshot_df.iloc[:,-n_classes:].apply(lambda x: np.argmax(x), axis=1)
            original_data['category'] = original_data.iloc[:,-n_classes:].apply(lambda x: np.argmax(x), axis=1)

            # create scatterplot
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(snapshot_df.loc[:,0], snapshot_df.loc[:,1], s=snapshot_df.loc[:,2]*200, c=snapshot_df.loc[:,'category'], cmap='brg', marker='o')
            ax.scatter(original_data.loc[:,0], original_data.loc[:,1], s=original_data.loc[:,2]*100, c=original_data.loc[:,'category'], cmap='brg', marker='P')
            ax.set_title('{}, epoch{}, error:{}'.format(problem, epoch+1, np.round(iteration_performance[epoch], 2)))

            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])

            # define paths and save
            output_dir = os.path.join(os.getcwd(), 'visuals')
            png = os.path.join(output_dir, 'gif_{}_{}.png'.format(problem, epoch))
            figure_file = ax.get_figure()
            figure_file.savefig(png)

    win.winHandle.activate() # needed because matplotlib steals focus

    return output_dir

def three_ft_gif(win, full_copy, problem, n_hids, wt_history, iteration_performance, n_classes):


    if n_classes == 2:

        for epoch in range(wt_history.shape[0]):

            # collect plot min/max
            combined_df = np.r_[full_copy[:,:3], wt_history[epoch,:,:3]]

            x_min = combined_df[:,0].min()
            x_max = combined_df[:,0].max()
            y_min = combined_df[:,1].min()
            y_max = combined_df[:,1].max()
            z_min = combined_df[:,2].min()
            z_max = combined_df[:,2].max()

            # remove extraneous categories, leaving only prob(A)
            snapshot = pd.DataFrame(wt_history[epoch,:,:-1])
            original_data = pd.DataFrame(np.copy(full_copy[:,:-1], order='K'))

            # upsize markers
            snapshot.loc[:,3] = snapshot.loc[:,3] * 200
            original_data.loc[:,3] = original_data.loc[:,3] * 200

            # create scatterplot
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(snapshot.loc[:,0], snapshot.loc[:,1], snapshot.loc[:,2], s=snapshot.loc[:,3]*2, c=snapshot.loc[:,4], cmap='RdBu', marker='o')
            ax.scatter(original_data.loc[:,0], original_data.loc[:,1], original_data.loc[:,2], s=original_data.loc[:,3], c=original_data.loc[:,4], cmap='RdBu', marker='P')
            ax.set_title('{}, epoch{}, error:{}'.format(problem, epoch+1, np.round(iteration_performance[epoch], 2)))

            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])
            ax.set_zticks([z_min, z_max])

            # define paths and save
            output_dir = os.path.join(os.getcwd(), 'visuals')
            png = os.path.join(output_dir, 'gif_{}_{}.png'.format(problem, epoch))
            figure_file = ax.get_figure()
            figure_file.savefig(png)
            plt.close('all')

    else:

        for epoch in range(wt_history.shape[0]):
            
            # collect plot min/max
            combined_df = np.r_[full_copy[:,:3], wt_history[epoch,:,:3]]

            x_min = combined_df[:,0].min()
            x_max = combined_df[:,0].max()
            y_min = combined_df[:,1].min()
            y_max = combined_df[:,1].max()
            z_min = combined_df[:,2].min()
            z_max = combined_df[:,2].max()

            # remove extraneous categories, leaving only prob(A)
            snapshot = pd.DataFrame(wt_history[epoch,:,:])
            original_data = pd.DataFrame(np.copy(full_copy[:,:]))

            # upsize markers
            snapshot.loc[:,3] = snapshot.loc[:,3] * 200
            original_data.loc[:,3] = original_data.loc[:,3] * 200

            # reverse the one-hot coding to get integer categories
            snapshot['category'] = snapshot.iloc[:,-n_classes:].apply(lambda x: np.argmax(x), axis=1)
            original_data['category'] = original_data.iloc[:,-n_classes:].apply(lambda x: np.argmax(x), axis=1)

            # create scatterplot
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(snapshot.loc[:,0], snapshot.loc[:,1], snapshot.loc[:,2], s=snapshot.loc[:,3], c=snapshot.loc[:,'category'], cmap='brg', marker='o')
            ax.scatter(original_data.loc[:,0], original_data.loc[:,1], original_data.loc[:,2], s=original_data.loc[:,3], c=original_data.loc[:,'category'], cmap='brg', marker='P')
            ax.set_title('{}, epoch{}, error:{}'.format(problem, epoch+1, np.round(iteration_performance[epoch], 2)))

            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])
            ax.set_zticks([z_min, z_max])

            # define paths and save
            output_dir = os.path.join(os.getcwd(), 'visuals')
            png = os.path.join(output_dir, 'gif_{}_{}.png'.format(problem, epoch))
            figure_file = ax.get_figure()
            figure_file.savefig(png)

    win.winHandle.activate() # needed because matplotlib steals focus

    return output_dir

def animate(output_dir, problem):

    # stitch together PNGs into mp4
    pngs = [img for img in os.listdir(output_dir) if img.startswith('gif')]
    pngs.sort(key=lambda img: os.path.getmtime(os.path.join(output_dir, img)))

    with imageio.get_writer(os.path.join(output_dir, '{}.gif'.format(problem)), fps=1) as writer:
        for png in pngs:
            writer.append_data(imageio.imread(os.path.join(output_dir, png)))

    # delete PNGs
    for delete_me in os.listdir(output_dir):
        delete_path = os.path.join(output_dir, delete_me)
        try:
            if delete_me.startswith('gif'):
                os.unlink(delete_path)
        except Exception as e:
            print(e)

## DICTIONARIES

run_fx = {'euclidean': [vanilla_warp, gen_weights, update_weights], 
          'dot product of differences': [warp_classic, gen_weights_cl, update_weights],
          'finite mixture': [cherry_warp, gen_weights_ch, update_weights_ch]}

key_names = {'comma':',',
            'period':'.',
            'minus':'-'}

gif_fx = {2:two_ft_gif,
          3:three_ft_gif}


