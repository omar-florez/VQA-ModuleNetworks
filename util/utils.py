import sys
##sys.path.append("/home/whyjay/caffe/python")
##sys.path.append("/usr/lib/python2.7/dist-packages/")

if sys.version_info[0] < 3: #check if version is less than 3
    import cv2 #not in python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import skimage
from skimage import io

import PIL.Image
import os
from PIL import Image

from scipy import ndimage
import re
import json

from collections import Counter

def open_image(x):
    image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)
    return image

def crop_image(x, target_height=227, target_width=227):
    image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width)/width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

def crop_image_PIL(x, target_height=227, target_width=227):
    data = np.asarray(Image.open(x))
    data = crop_image_PIL_data(data, target_height=target_height, target_width=target_width)
    return data

def crop_image_PIL_data(data, target_height=227, target_width=227):
    if len(data.shape) == 2:
        data = np.tile(data[:,:,None], 3)
    elif len(data.shape) == 4:
        data = data[:,:,:,0]

    img = Image.fromarray(np.asarray(data, np.uint8))
    height, width, rgb = data.shape

    if width == height:
        resized_image = np.asarray(img.resize((target_height, target_width), PIL.Image.ANTIALIAS))
    elif height < width:
        img = img.resize((int(width * float(target_height) / height), target_width), PIL.Image.ANTIALIAS)
        resized_image = np.asarray(img)
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]
    else:
        img = img.resize((target_height, int(height * float(target_width)/width)), PIL.Image.ANTIALIAS)
        resized_image = np.asarray(img)
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    img = Image.fromarray(resized_image)
    img = img.resize((target_height, target_width), PIL.Image.ANTIALIAS)
    data = np.array(img)
    return data

def build_word_vocab(sentence_iterator, word_count_threshold=10): # borrowed this function from NeuralTalk
    print( 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def shuffle_block_df(df):
    index = list(df.index)
    block_idx = range(len(index)/5)
    np.random.shuffle(block_idx)

    new_index = []
    for b in block_idx:
        new_index += index[5*b : 5*(b+1)]
    df = df.ix[new_index]
    return df.reset_index(drop=True)

def shuffle_df(df):
    index = list(df.index)
    np.random.shuffle(index)
    df = df.ix[index]
    return df.reset_index(drop=True)

#def prep_cocoeval_flickr(ann_df, res_df):
#    pass

# ann_df uniq image images
#ann = {'images':None, 'info':'', 'type':'captions', 'annotations':None}
#ann_caps = {'caption':"afjiwel", 'id':1, 'image_id':2}
#ann_images = {'id':2}
#res = [{'caption':'hello', 'image_id':2}, {}]
# return metric

def get_mostcommon_answer(answer_list):
    #given the Q&A corresponding to a question, return the most frequent answer
    lst = []
    for answer in answer_list:
        lst.append(answer['answer'])
    return most_common(lst)

def get_set_answer(answer_list):
    #given the Q&A corresponding to a question, return the set of answers
    lst = []
    for answer in answer_list:
        lst.append(answer['answer'])
    return np.unique(lst)

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

#-------------------------------------------------------------------------------
#  Image auxiliary functions
#-------------------------------------------------------------------------------
def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = max_size / np.max(image.size)
    
        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS)

    # Convert to numpy floating-point array.
    return np.float32(image)

def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    
    # Convert to bytes.
    image = image.astype(np.uint8)
    
    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')
        
def plot_image_big(image):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert pixels to bytes.
    image = image.astype(np.uint8)

    # Convert to a PIL-image and display it.
    display(PIL.Image.fromarray(image))
    
def plot_images(content_image, style_image, mixed_image, savefile=False, 
                epoch=0, image_name="", save_folder=""):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True
    
    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")

    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    if savefile:
        plt.savefig('%s/content_mixed_style-%s-%d.png' %(save_folder, image_name, epoch))
    plt.show()

def plot_accuracy_loss(input_file_path, output_file_path = './', title='results'):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10))
    #fig.subplots_adjusts(hspace=0.1, wspace=0.1)

    with open(input_file_path) as file:
        content = file.readlines()
        dataset = [x.split(',') for x in content]

    dataset = np.matrix(dataset).astype(float)
    #ipdb.set_trace()
    epoch = np.array(dataset[:, 0])
    loss_train = np.array(dataset[:, 1])
    loss_val = np.array(dataset[:, 2])
    accuracy_train = np.array(dataset[:, 3])
    accuracy_val = np.array(dataset[:, 4])

    ax1.plot(epoch, loss_train, color='b', label="train")
    ax1.plot(epoch, loss_val, color='r', label="validation")
    ax1.scatter(epoch, loss_train, color='b')
    ax1.scatter(epoch, loss_val, color='r')
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss value")
    ax1.legend(loc=2, borderaxespad=0.)
    ax1.set_title(title)

    ax2.plot(epoch, accuracy_train, color='b', label="train")
    ax2.plot(epoch, accuracy_val, color='r', label="validation")
    ax2.scatter(epoch, accuracy_train, color='b')
    ax2.scatter(epoch, accuracy_val, color='r')
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("accuracy")
    ax2.legend(loc=2, borderaxespad=0.)

    plt.savefig(output_file_path)

    
def create_folders(folder_name):
    if not os.path.isdir(folder_name) and (folder_name!='.' or folder_name!='./'):
        os.makedirs(folder_name)


def plot_single_inference_old(content_image, attention1, attention2, savefile=False, 
                          image_id=0, title="aa", save_folder="."):
    
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(50, 50))
    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    # Use interpolation to smooth pixels?
    smooth = True
    
    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(attention1 / 255.0, interpolation=interpolation)
    ax.set_xlabel("Attention 1")

    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(attention2 / 255.0, interpolation=interpolation)
    ax.set_xlabel("Attention 2")

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(title, size=100)
    if savefile:
        create_folders(save_folder)
        plt.savefig('%s/vqa_inference-%d.png' %(save_folder, image_id))
    else:
        plt.show()
        
def plot_single_inference(content_image, attention1, savefile=False, 
                          image_id=0, title="", save_folder=".",
                          file_path=''):
    attention1 = np.reshape(attention1, [14, 14])
    # Create figure with sub-plots.
    #fig, axes = plt.subplots(1, 2, figsize=(50, 50), gridspec_kw = {'width_ratios':[3, 1]})
    fig, axes = plt.subplots(1, 3, figsize=(50, 50))
    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    # Use interpolation to smooth pixels?
    smooth = True
    
    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(attention1 / 255.0, interpolation=interpolation)
    ax.set_xlabel("Attention 1")


    # Plot alpha superpuesto
    img = ndimage.imread(file_path)
    ax = axes.flat[2]
    
    #attention1_resized = attention1.resize((new_width, new_height), Image.ANTIALIAS)
    
    ax.imshow(attention1 / 255.0, interpolation=interpolation)
    ax.imshow(content_image, interpolation=interpolation, alpha=0.6)
    ax.set_xlabel("Content")
                    
    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(title, size=100)
    if savefile:
        create_folders(save_folder)
        plt.savefig('%s/vqa_inference-%d.png' %(save_folder, image_id))
    else:
        plt.show()


def plot_multiple_inference(content_image,
                            image_feature_raw,
                            image_feature,
                            h_att,
                            out_att,
                            alpha_att,
                            context,
                            h_att_context,
                            savefile=False, image_id=0, title="", save_folder=".", file_path=''):
    matplotlib.rcParams.update({'font.size': 60})
    fig, axes = plt.subplots(3, 4, figsize=(50, 50))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels
    smooth = True
    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'
    alpha_v = 0.9

    #------------------------------------------
    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(crop_image_PIL_data(content_image) / 255.0, interpolation=interpolation)
    ax.set_title("Content")
    ax = axes.flat[1]
    viz_map = np.reshape(np.mean(h_att_context, 1), [14, 14])
    ax.imshow(crop_image_PIL_data(content_image) / 255.0, interpolation=interpolation, alpha=1.0)
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation, alpha=alpha_v)
    ax.set_title("h_att_context")

    ax = axes.flat[2]
    viz_map = np.reshape(alpha_att, [14, 14])
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation)
    ax.set_title("alpha_att")
    ax = axes.flat[3]
    viz_map = np.reshape(alpha_att, [14, 14])
    ax.imshow(crop_image_PIL_data(content_image) / 255.0, interpolation=interpolation, alpha=1.0)
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation, alpha=alpha_v)
    ax.set_title("alpha_att")

    #------------------------------------------
    #image_feature_raw
    ax = axes.flat[4]
    viz_map = np.reshape(np.mean(image_feature_raw, 0), [14,14])
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation)
    ax.set_title("image_feature_raw")
    ax = axes.flat[5]
    viz_map = np.reshape(np.mean(image_feature_raw, 0), [14, 14])
    ax.imshow(crop_image_PIL_data(content_image) / 255.0, interpolation=interpolation, alpha=1.0)
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation, alpha=alpha_v)
    ax.set_title("image_feature_raw")

    #------------------------------------------
    #image_feature
    ax = axes.flat[6]
    viz_map = np.reshape(np.mean(image_feature, 1), [14,14])
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation)
    ax.set_title("image_feature")
    ax = axes.flat[7]
    viz_map = np.reshape(np.mean(image_feature, 1), [14, 14])
    ax.imshow(crop_image_PIL_data(content_image) / 255.0, interpolation=interpolation, alpha=1.0)
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation, alpha=alpha_v)
    ax.set_title("image_feature")

    #------------------------------------------
    #h_att
    ax = axes.flat[8]
    viz_map = np.reshape(np.mean(h_att, 1), [14,14])
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation)
    ax.set_title("h_att")

    ax = axes.flat[9]
    viz_map = np.reshape(np.mean(h_att, 1), [14, 14])
    ax.imshow(crop_image_PIL_data(content_image) / 255.0, interpolation=interpolation, alpha=1.0)
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation, alpha=alpha_v)
    ax.set_title("h_att")

    #out_att
    ax = axes.flat[10]
    viz_map = np.reshape(out_att, [14,14])
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation)
    ax.set_title("out_att")
    ax = axes.flat[11]
    viz_map = np.reshape(out_att, [14,14])
    ax.imshow(crop_image_PIL_data(content_image) / 255.0, interpolation=interpolation, alpha=1.0)
    ax.imshow(crop_image_PIL_data(alpha_viz(viz_map)), interpolation=interpolation, alpha=alpha_v)
    ax.set_title("out_att")

    # ------------------------------------------

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, size=100)
    if savefile:
        create_folders(save_folder)
        plt.savefig('%s/vqa_inference-%d.png' % (save_folder, image_id))
    else:
        plt.show()

    return

def stats(x): print('Min: %f Max: %f Mean: %f' %(np.min(x), np.max(x), np.mean(x)))

def plot_1_image(content_image, mask, name='1file', alpha1=1.0, alpha2=0.9):
    matplotlib.rcParams.update({'font.size': 60})
    fig, axes = plt.subplots(1, 2, figsize=(50, 50))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    smooth = True
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    ax = axes.flat[0]
    ax.imshow(crop_image_PIL_data(content_image) / 255.0, interpolation=interpolation, alpha=alpha1)
    ax.set_title("content")

    ax = axes.flat[1]
    ax.imshow(crop_image_PIL_data(content_image) / 255.0, interpolation=interpolation, alpha=alpha1)
    ax.imshow(crop_image_PIL_data(mask), interpolation=interpolation, alpha=alpha2)
    ax.set_title(name)
    plt.savefig('./vqa_inference-%s.png' % (name))
    plt.close()

def znormalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x

def standarize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def alpha_viz(x):
    #x = np.abs(x);
    x = (1.0 - standarize(x))*255.0
    return x

def alpha_viz_old(x):
    x = np.max(np.abs(x), 2);
    x = (1.0 - standarize(x))*255.0
    return x


#https://gist.github.com/stober/1946926
def softmax(x):
    '''
    >>> res = softmax(np.array([0, 200, 10]))
    >>> np.sum(res)
    1.0
    >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
    True
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.,  1.])
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.])
    '''
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

def precision(x, y):
    '''x and y are two 1D arrays'''
    if len(x)==0 or len(y)==0:
        return -1
    y = np.array(y)
    x = np.array(x)
    hits = np.sum(x==y)
    return float(hits)/len(x)*100.0

def overlap_images(image_path1, image_path2, target_height, target_width):
    img1 = crop_image_PIL(image_path1, target_height=target_height, target_width=target_width)
    img2 = crop_image_PIL(image_path2, target_height=target_height, target_width=target_width)

    fig, axes = plt.subplots(1, 3, figsize=(50, 50))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    smooth = True
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    ax = axes.flat[0]
    #ax.imshow(img1, interpolation=interpolation)
    ax.imshow(img1)

    ax.set_title("img1")

    ax = axes.flat[1]
    #ax.imshow(img2, interpolation=interpolation)
    ax.imshow(img2)
    ax.set_title("img2")

    ax = axes.flat[2]
    ax.imshow(img1, interpolation=interpolation)
    ax.imshow(img2, interpolation=interpolation, alpha=0.6)
    ax.set_title("img1+img2")
    plt.savefig('%s/vqa_inference-%s.png' % ('.', 'alpha_image'))

#Visualize update of gradients based on reward
#   episode_hs:         hidden embedding before gradient updates (N,D)
#   post_episode_hs:    hidden embedding after gradient updates (N,D)
#   episode_rewards:    rewards obtained during episode (N,1)
#   episode_number:     to get file name or title
def plot_gradient_estimator(episode_hs, post_episode_hs, episode_rewards, episode_number=0):
    matplotlib.rcParams.update({'font.size': 60})
    n_components=2

    print('np.mean(episode_hs): ', np.mean(episode_hs))
    print('np.mean(post_episode_hs): ', np.mean(post_episode_hs))
    print('np.mean(episode_rewards): ', np.mean(episode_rewards))
    #episode_hs -= np.mean(episode_hs)
    #episode_hs /= np.std(episode_hs)
    #post_episode_hs -= np.mean(post_episode_hs)
    #post_episode_hs /= np.std(post_episode_hs)

    if np.any(np.isnan(episode_hs)) or np.any(np.isnan(post_episode_hs)):
        return

    episode_hs_pca = PCA_projection(np.transpose(episode_hs), n_components)
    post_episode_pca = PCA_projection(np.transpose(post_episode_hs), n_components)

    fig, axes = plt.subplots(1, 2, figsize=(50, 50))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    red = (1, 0, 0)
    green = (0, 1, 0)
    area = 200

    ax = axes.flat[0]
    colors = np.array([red for x in episode_rewards])
    index = np.where(np.array(episode_rewards>0))[0]
    if len(index)>0:
        colors[index] = green
        avg_distance_before = compute_average_distance(episode_hs_pca[index], episode_hs_pca)
        avg_distance_after = compute_average_distance(post_episode_pca[index], post_episode_pca)

    ax.scatter(episode_hs_pca[:,0], episode_hs_pca[:,1], s=area, c=colors, alpha=0.7)
    ax.scatter(episode_hs_pca[index,0], episode_hs_pca[index,1], s=500, c=colors[index], alpha=0.7)
    ax.set_title("Average distance: %.3f\nBefore update"%avg_distance_before)

    ax = axes.flat[1]
    ax.scatter(post_episode_pca[:, 0], post_episode_pca[:, 1], s=area, c=colors, alpha=0.5)
    plt.quiver(episode_hs_pca[:,0], episode_hs_pca[:,1], post_episode_pca[:,0], post_episode_pca[:,1], units='width')
    ax.scatter(post_episode_pca[index, 0], post_episode_pca[index, 1], s=900, c=colors[index], alpha=0.7)
    ax.set_title("Average distance: %.3f\nAfter update" % avg_distance_after)

    plt.savefig('%s/gradient_update-%d.png' % ('.', episode_number))

def analyze_expressions(tree_expressions_output_file, output_folder=".", output_file='assembled_models'):
    with open(tree_expressions_output_file, 'r') as f:
        web_content = np.array(json.load(f))
        expr_list = []

        #format string to be able to count
        for x in web_content:
            #x = str(x).replace("'", "\"")
            ##ipdb.set_trace()
            x = str(x)
            x = re.sub(r"\'question\': \".*\"*\'*.*?\"", "'question': \"\"", x)
            x = re.sub(r"\'question\': \'.*?\'", "'question': \"\"", x)
            x = re.sub(r"\'batch_idx\': [0-9]+", "'batch_idx': 0", x)
            expr_list.append(x)

        #counter
        type_questions_dic= {}
        index_questions_dic = {}
        for i, x in enumerate(expr_list):
            type_questions_dic.setdefault(x, []).append(x)
            index_questions_dic.setdefault(x, []).append(i)

        type_questions_keys = [x for x in type_questions_dic.keys()]
        #type_questions_count = [len(type_questions_dic[x]) for x in type_questions_dic.keys() if len(type_questions_dic[x])>5]
        type_questions_count = [len(type_questions_dic[x]) for x in type_questions_dic.keys()]
        type_questions_indices = [index_questions_dic[x] for x in type_questions_keys]

        #plot count histograms for questions
        index = np.arange(len(type_questions_count))
        plt.barh(index, type_questions_count, alpha=0.7)
        plt.ylabel('Model generated')
        plt.xlabel('Frequency')
        plt.title("Frequency of models assembled")
        plt.yticks(index, ['Model %d (%i)'%(i, type_questions_count[i])  for i in index])
        plt.tight_layout()
        #plt.margins(1.2)
        plt.savefig('%s/%s_count.png' % (output_folder, output_file))

        #save model types:
        with open('%s/%s_types.json' % (output_folder, output_file), 'w') as f:
            print('Writing model types.')
            json.dump(type_questions_keys, f, ensure_ascii=False, indent = 4)
        # save 10 questions for each model type:
        dict_model_questions = {}
        with open('%s/%s_10questions.json' % (output_folder, output_file), 'w') as f:
            print('Writing model corresponding questions.')
            top10_questions_per_model=[]
            key = "model_%d_%d"
            for ii, indices in enumerate(type_questions_indices):
                k = key%(ii, len(indices))
                if len(indices)>1:
                    v = [content['question'] for content in web_content[indices[:10]]]
                else:
                    v = [web_content[indices][0]['question']]
                top10_questions_per_model.append({k: v})
            json.dump(top10_questions_per_model, f, ensure_ascii=False, indent = 4)


if __name__ == '__main__':
    #file_in = './accuracy/concat_local.lr_0.0005-dropout_0.6-tanh_dropout_image2.csv'
    #file_out = './accuracy/concat_local.lr_0.0005-dropout_0.6-tanh_dropout_image2.png'
    #title = 'concat_local.lr_0.0005-dropout_0.5-batchnorm_tanh_dropout_image'
    #plot_accuracy_loss(file_in, file_out, title=title)

    #--
    from skimage import io
    #x = '/home/ouflorez/workspace/VQA2015/mscoco/visual_data/images/train2014/COCO_train2014_000000155172.jpg'

    image_path1 = '/Users/ouflorez/WorkspaceLiClipse/VQA2015/visual_data/mscoco/images/val2014/COCO_val2014_000000000042.jpg'
    image_path2 = '/Users/ouflorez/WorkspaceLiClipse/VQA2015/visual_data/mscoco/images/val2014/COCO_val2014_000000000073.jpg'
    overlap_images(image_path1, image_path2, target_height=50, target_width=50)

