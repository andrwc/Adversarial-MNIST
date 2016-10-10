import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64

"""General analysis and some viz"""

def get_adversarial_sixes(orig, adver, idxes):
    """(np.array, np.array, list) -> (np.array, np.array)
    Return a tuple containing the original mnist and adversarial mnist images,
    respectively, that were found to be classified as the digit six with high
    confidence

    :param orig: the original mnist images for the digit 2.
    :param adver: the adversarial mnist images for the digit 2.
    """
    return orig[idxes], adver[idxes]

def plot_images(preds):
    """(np.array) -> None"""
    preds['argmax'] = preds.T.apply(lambda x: np.argmax(x))
    preds['argmax'].plot(kind='hist', alpha=0.5, title="Predicted class for adversarial 2's")
    plt.savefig('class_frequencies.png', dpi=100)
    plt.close()
    del preds['argmax']
    preds.apply(lambda x: x.plot(kind='hist', bins=100, stacked=True, legend=True, title="Per-class Hist of (Unnormalized) Confidence Scores for Adversarial 2's", alpha=0.5))
    plt.savefig('hist.png', dpi=100)
    plt.close()
    preds.plot.box()
    plt.xlabel("Class")
    plt.ylabel("Confidence Score")
    plt.savefig('box.png', dpi=100)
    plt.close()
    return

if __name__=='__main__':


    origtwos = np.load('original_twos.npy')
    adtwos = np.load('adversarial_twos.npy')
    adtwos_pred = np.load('adversarial_twos_pred.npy')

    preds = pd.DataFrame(adtwos_pred)
    plot_images(preds)

    predclasses = pd.DataFrame(np.argmax(adtwos_pred, axis=1))
    sixes_idx = predclasses[predclasses[0]==6].index

    predscore = pd.Series(np.max(adtwos_pred[sixes_idx], axis=1))
    predscore.sort(ascending=False) # inplace

    orig, adv = get_adversarial_sixes(origtwos, adtwos, sixes_idx)

    def wrapdiv(x, style=None):
        return '<div>{}</div>'.format(x)

    def wrapspan(x):
        return '<span>{}</span>'.format(x)

    html = []
    html.append("<!doctype html><body>")
    html.append(wrapdiv("<p>On the left, the original image. In the middle, the \
        adversarial image that our deep CNN incorrectly classifies as a 6. On \
        the right, the (unnormalized) confidence score the model assigned to \
        the adversarial example.</p>"))
    imtmpl = "<img src=\"data:image/png;base64, {}\"></img>"
    tmpl = "images/{}.png"
    tmpl_adv = "images/{}adv.png"
    for i in predscore.index: # descending wrt confidence "as a six"
        o_fp = tmpl.format(i)
        o_fp_adv = tmpl_adv.format(i)

        mpimg.imsave(o_fp, orig[i].reshape(28,28), cmap=plt.cm.binary)
        mpimg.imsave(o_fp_adv, adv[i].reshape(28,28), cmap=plt.cm.binary)

        with open(o_fp, 'r') as fh:
            data = fh.read()
            b64 = data.encode('base64')
            o_im = imtmpl.format(b64)

        with open(o_fp_adv, 'r') as fh:
            data = fh.read()
            b64 = data.encode('base64')
            o_im_adv = imtmpl.format(b64)

        scr = wrapspan(predscore.ix[i])

        html.append(wrapdiv(o_im + o_im_adv + scr))

    html.append("</body>")

    with open('comparison.html', 'w') as fh:
        fh.write("\n".join(html))
        
