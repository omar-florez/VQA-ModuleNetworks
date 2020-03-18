from __future__ import absolute_import, division, print_function

import os
import sys
import threading
import queue
import numpy as np
import subprocess
import re

#load CNN module when running on Python2
if not sys.version_info > (3, 3):
    from exp_vqa.data.cnn import *


from util import text_processing



layer_set = {
    'res4b22_branch2b_train2014' : {'layers'      : 'res4b22_branch2b',
                          'layer_size'  : [256,14,14],
                          'images_path' : '/home/ouflorez/workspace/VQA2015/mscoco/visual_data/images/train2014/',
                          'feat_path'   : '/home/ouflorez/workspace/n2nmn/exp_vqa/data/resnet_res5c/train2014/'},
    'res4b22_branch2b_val2014': {'layers': 'res4b22_branch2b',
                               'layer_size': [256, 14, 14],
                               'images_path': '/home/ouflorez/workspace/VQA2015/mscoco/visual_data/images/val2014/',
                               'feat_path': '/home/ouflorez/workspace/n2nmn/exp_vqa/data/resnet_res5c/val2014/'},
    'res4b22_branch2b_test2015': {'layers': 'res4b22_branch2b',
                                 'layer_size': [256, 14, 14],
                                 'images_path': '/home/ouflorez/workspace/VQA2015/mscoco/visual_data/images/test2015/',
                                 'feat_path': '/home/ouflorez/workspace/n2nmn/exp_vqa/data/resnet_res5c/test2015/'},
    'res4b22_branch2b_demo': {'layers': 'res4b22_branch2b',
                                'layer_size': [256, 14, 14],
                                'images_path': '/home/ouflorez/workspace/VQA2015/demo/visual_data/images/',
                                'feat_path': '/home/ouflorez/workspace/VQA2015/demo/visual_data/features/',
                                'output_path': '/home/ouflorez/workspace/VQA2015/demo/visual_data/output/',
                                'target_height': 224,
                                'target_width': 224}
}

class QueryReader:
    def __init__(self, **kwargs):
        print('Creating query')
        self.batch_size = kwargs['batch_size']
        self.T_encoder = kwargs['T_encoder']
        self.T_decoder = kwargs['T_decoder']
        self.assembler = kwargs['assembler']
        self.vocab_question_file = kwargs['vocab_question_file']
        self.vocab_answer_file = kwargs['vocab_answer_file']
        self.vocab_dict = text_processing.VocabDict(self.vocab_question_file)
        self.answer_dict = text_processing.VocabDict(self.vocab_answer_file)

    def extract_cnn_features(self, network_dic, image_paths):
        PROJECT_HOME = '/home/ouflorez/workspace/VQA2015'  # /home/ouflorez/workspace/VQA2015
        res_model = os.path.join(PROJECT_HOME, 'models/resnet/ResNet-101-model.caffemodel')
        res_deploy = os.path.join(PROJECT_HOME, 'models/resnet/ResNet-101-deploy.prototxt')

        cnn = CNN(model=res_model, deploy=res_deploy, width=network_dic['target_width'],
                  height=network_dic['target_height'], batch_size=1)

        layers = network_dic['layers']
        layer_size = network_dic['layer_size']
        feat_path = network_dic['feat_path']

        if not os.path.exists(network_dic['feat_path']):
            os.makedirs(network_dic['feat_path'])

        print('...Start CNN encoding')
        #feats: (1, 256, 14, 14)
        feats = cnn.get_features(image_paths,
                                 layers=layers,
                                 layer_sizes=layer_size)
        assert len(feats) == len(image_paths), "Different number of files and features"

        print("Number of observations: ", len(feats))
        for i in range(len(feats)):
            file_name = os.path.basename(image_paths[i]).split('.')[0]
            feat = feats[i]
            np.save(os.path.join(feat_path, file_name + '.npy'), feat)
        print("Extract CNN embedding")
        return feats


    def build_query_batch(self, question, image_path, network_dic=layer_set['res4b22_branch2b_test2015']):
        #image encoding
        image_paths = [image_path]
        actual_batch_size=len(image_paths)

        # python2
        if not sys.version_info > (3, 3):
            feats = self.extract_cnn_features(network_dic, image_paths)
        else:
            input_file = image_path
            output_file = './exp_vqa/data/tmp.npy'
            os.makedirs(os.path.split(output_file)[0], exist_ok=True)
            command = "python2 exp_vqa/data/cnn.py %s %s" %(input_file, output_file)
            return_code = subprocess.call(command, shell=True)
            feats = np.load(output_file)

        self.feat_D, self.feat_H, self.feat_W = feats.shape[1:]

        input_seq_batch = np.zeros((self.T_encoder, actual_batch_size), np.int32)
        seq_length_batch = np.zeros(actual_batch_size, np.int32)
        image_feat_batch = np.zeros((actual_batch_size, self.feat_H, self.feat_W, self.feat_D), np.float32)
        image_path_list = [None] * actual_batch_size
        qstr_list = [None] * actual_batch_size

        #tokenize question:
        SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
        question_tokens = SENTENCE_SPLIT_REGEX.split(question.lower())
        question_tokens = [t.strip() for t in question_tokens if len(t.strip()) > 0]

        #build batch
        n=0
        question_inds = [self.vocab_dict.word2idx(w) for w in question_tokens]
        seq_length = len(question_inds)
        input_seq_batch[:seq_length, n] = question_inds
        seq_length_batch[n] = seq_length

        image_feat_batch[n:n + 1] = feats[0].transpose([1,2, 0])
        batch = dict(input_seq_batch = input_seq_batch,
                     seq_length_batch = seq_length_batch,
                     image_feat_batch = image_feat_batch,
                     image_path_list = image_path_list,
                     qid_list = -1, qstr_list=-1)
        return batch

if __name__ == '__main__':
    assembler = None
    # Data files
    # ['_Scene', '_Find', '_Filter', '_FindSameProperty', '_Transform', '_And', '_Or', '_Exist', '_Count', '_EqualNum',
    # '_MoreNum', '_LessNum', '_SameProperty', '_Describe', '<eos>']
    vocab_layout_file = './exp_vqa/data/vocabulary_layout.txt'
    vocab_answer_file = './exp_vqa/data/answers_vqa.txt'
    vocab_question_file = './exp_vqa/data/vocabulary_vqa.txt'

    query_reader = QueryReader(batch_size=1, T_encoder=30, T_decoder=3, assembler=assembler,
                               vocab_question_file=vocab_question_file, vocab_answer_file=vocab_answer_file)

    question = 'How many people are in the picture?'
    image_path = '/home/ouflorez/workspace/VQA2015/demo/visual_data/images/COCO_train2014_000000005619.jpg'
    network_dic=layer_set['res4b22_branch2b_demo']
    batch = query_reader.build_query_batch(question, image_path, network_dic=network_dic)
