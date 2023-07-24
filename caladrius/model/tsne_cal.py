#Todo: irrelevant
from sklearn.manifold import TSNE
import numpy as np

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import torch
import copy

def run_tsne(qsn, datasets):
    '''
    Inputs:
    - qsn: A siamese convolutional neural network from 510
    - datasets: a Datasets object from 510 caladrius code

    outputs:
    - tsne decomposition
    - labels for this respective decomposition
    '''
    train_set, train_loader = datasets.load("train", active_inference = True) #active_inference = True added such that all images are processed as drop_last is set to False
    activation = {}
    def get_activation(name): #to return the right activation from the model; works by registering hook on wanted layer; see https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    qsn.model.similarity.register_forward_hook(get_activation('similarity')) #register hook to similarity layer

    feat_lab = []
    for idx, (filename, image1, image2, labels) in enumerate(train_loader): #run for batches of images in train_loader
        if torch.cuda.is_available(): #Run model such that featrues can be extracted with hook; use cuda when GPU available, cpu else
            output = qsn.model(image1.cuda(), image2.cuda())
        else:
            output = qsn.model(image1.cpu(), image2.cpu())
        features = activation['similarity'] #Extract features
        feat_lab.append((features.detach().cpu().numpy(), labels.detach().cpu().numpy().astype(int))) #store features together with labels
    all_features, labels = zip(*feat_lab) #extract features and labels; after this they are concatenated in nump arrays
    all_features2 = np.vstack(all_features)
    labels = np.concatenate(labels)

    tsne = TSNE(n_jobs=4)
    tsne_decomposition = tsne.fit_transform(all_features2) #Run TSNE decomposition; Default metric Euclidean distance is used.

    return tsne_decomposition, labels


def make_animation(path, frames):
    '''
    input:
    - path: output path
    - frames:
    output: Saves animation of labelled sets of images using tsne decomposition over active iterations to output path
            in both html version (interactive with play/pause button) and gif


    '''
    def plot_images(img_list):
        def init():
            img.set_data(img_list[0])
            return (img,)

        def animate(i):
            img.set_data(img_list[i])
            return (img,)

        fig = plt.Figure(figsize=(10, 10))
        ax = fig.gca()
        img = ax.imshow(img_list[0])
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(img_list), interval=60, blit=True)
        return anim
    anim = plot_images(frames)
    anim.save(os.path.join(path,"animation_labelling_tsne.gif"), writer=animation.PillowWriter(fps=2)) #write to a gif; change fps after creation with: https://ezgif.com/speed
    animat_fig_html = HTML(anim.to_jshtml())
    with open(os.path.join(path,"animation_labelling_tsne.html"), "w") as file: #Write to html
        file.write(animat_fig_html.data)





