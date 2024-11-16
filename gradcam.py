# from A3

import torch
import torchvision
from torchvision import transforms as T
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import GuidedGradCam, GuidedBackprop
from captum.attr import LayerActivation, LayerConductance, LayerGradCam

from data_utils import *
from image_utils import *
from captum_utils import *
import numpy as np
import os

# from A3
SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)
def load_imagenet_val(num=None):
    """Load a handful of validation images from ImageNet.

    Inputs:
    - num: Number of images to load (max of 25)

    Returns:
    - X: numpy array with shape [num, 224, 224, 3]
    - y: numpy array of integer image labels, shape [num]
    - class_names: dict mapping integer label to class name
    """
    imagenet_fn = 'data/imagenet_val_25.npz'
    if not os.path.isfile(imagenet_fn):
      print('file %s not found' % imagenet_fn)
      print('Run the following:')
      print('cd cs7643/datasets')
      print('bash get_imagenet_val.sh')
      assert False, 'Need to download imagenet_val_25.npz'
    f = np.load(imagenet_fn, allow_pickle=True)
    X = f['X']
    y = f['y']
    class_names = f['label_map'].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names
def visualize_attr_maps(path, X, y, class_names, attributions, titles, attr_preprocess=lambda attr: attr.permute(1, 2, 0).detach().numpy(),
                        cmap='viridis', alpha=0.7):
    '''
    A helper function to visualize captum attributions for a list of captum attribution algorithms.

    path (str): name of the final saved image with extension (note: if batch of images are in X,
                      all images/plots saved together in one final output image with filename equal to path)
    X (numpy array): shape (N, H, W, C)
    y (numpy array): shape (N,)
    class_names (dict): length equal to number of classes
    attributions(A list of torch tensors): Each element in the attributions list corresponds to an
                      attribution algorithm, such an Saliency, Integrated Gradient, Perturbation, etc.
    titles(A list of strings): A list of strings, names of the attribution algorithms corresponding to each element in
                      the `attributions` list. len(attributions) == len(titles)
    attr_preprocess: A preprocess function to be applied on each image attribution before visualizing it with
                      matplotlib. Note that if there are a batch of images and multiple attributions
                      are visualized at once, this would be applied on each infividual image for each attribution
                      i.e attr_preprocess(attributions[j][i])
    '''
    N = attributions[0].shape[0]
    plt.figure()
    for i in range(N):
        axs = plt.subplot(len(attributions) + 1, N + 1, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])

    plt.subplot(len(attributions) + 1, N + 1, N + 1)
    plt.text(0.0, 0.5, 'Original Image', fontsize=14)
    plt.axis('off')
    for j in range(len(attributions)):
        for i in range(N):
            plt.subplot(len(attributions) + 1, N + 1, (N + 1) * (j + 1) + i + 1)
            attr = np.array(attr_preprocess(attributions[j][i]))
            attr = (attr - np.mean(attr)) / np.std(attr).clip(1e-20)
            attr = attr * 0.2 + 0.5
            attr = attr.clip(0.0, 1.0)
            plt.imshow(attr, cmap=cmap, alpha=alpha)
            plt.axis('off')
        plt.subplot(len(attributions) + 1, N + 1, (N + 1) * (j + 1) + N + 1)
        plt.text(0.0, 0.5, titles[j], fontsize=14)
        plt.axis('off')

    plt.gcf().set_size_inches(20, 13)
    plt.savefig(path, bbox_inches = 'tight')
    # plt.show()

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load flood images from Sen1Floods11/training notebook
X, y, class_names = load_imagenet_val(num=5)


# **************************************************************************************** #
# Captum
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

# Convert X and y from numpy arrays to Torch Tensors
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)

conv_module = model.features[12]

##############################################################################
# TODO: Compute/Visualize GuidedBackprop and Guided GradCAM as well.         #
#       visualize_attr_maps function from captum_utils.py is useful for      #
#       visualizing captum outputs                                           #
#       Use conv_module as the convolution layer for gradcam                 #
##############################################################################
# Computing Guided GradCam
guided_gc = GuidedGradCam(model, conv_module)
attr_ggc = guided_gc.attribute(X_tensor, y_tensor)
visualize_attr_maps('visualization/guided_gradcam_captum', X, y, class_names, [attr_ggc], ['Guided GradCam'])

# Computing Guided BackProp
guided_bp = GuidedBackprop(model)
attr_gbp = guided_bp.attribute(X_tensor, y_tensor)
visualize_attr_maps('visualization/guided_backprop_captum', X, y, class_names, [attr_gbp], ['Guided Backprop'])
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

# Try out different layers and see observe how the attributions change

layer = model.features[3]

# Example visualization for using layer visualizations 
# layer_act = LayerActivation(model, layer)
# layer_act_attr = compute_attributions(layer_act, X_tensor)
# layer_act_attr_sum = layer_act_attr.mean(axis=1, keepdim=True)


##############################################################################
# TODO: Visualize Individual Layer Gradcam and Layer Conductance (similar    #
# to what we did for the other captum sections, using our helper methods),   #
# but with some preprocessing calculations.                                  #
#                                                                            #
# You can refer to the LayerActivation example above and you should be       #
# using 'layer' given above for this section                                 #
#                                                                            #
# For layer gradcam look at the usage of the parameter relu_attributions.    #
# Also, note that Layer gradcam aggregates across all channels (Refer to     #
# Captum docs)                                                               #
##############################################################################
layer_gc = LayerGradCam(model, layer)
layer_gc_attr = layer_gc.attribute(X_tensor, y_tensor, relu_attributions=True)
visualize_attr_maps('visualization/layer_gradcam_captum', X, y, class_names, [layer_gc_attr], ['Individual Layer GradCam'])
layer_cond = LayerConductance(model, layer)
layer_cond_attr = layer_cond.attribute(X_tensor, target=y_tensor)
layer_cond_attr_sum = layer_cond_attr.sum(dim=1, keepdim=True)
visualize_attr_maps('visualization/layer_conductance_captum', X, y, class_names, [layer_cond_attr_sum], ['Individual Layer Conductance'])
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

