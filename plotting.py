import random
import numpy as  np
import matplotlib.pyplot as plt


def plot_sample(X, y,binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix].squeeze(), cmap='gray')
    ax[0].set_title('Input Image')

    ax[1].imshow(y[ix].squeeze(), cmap='gray')
    ax[1].set_title('Ground Truth')

    ax[2].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1, cmap='gray')
    ax[2].set_title('pred')

    ax[3].imshow(y[ix].squeeze(), cmap='autumn', )
    ax[3].imshow(np.ma.masked_where(binary_preds[ix].squeeze() == False, binary_preds[ix].squeeze()), cmap='bone',
                 alpha=0.6)
    ax[3].set_title('prediction on ground truth')