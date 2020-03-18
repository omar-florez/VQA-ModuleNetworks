#nmn3_model.py

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_fold as td
from tensorflow import convert_to_tensor as to_T
import ipdb

from models_vqa.nmn3_netgen_att import AttentionSeq2Seq
from models_vqa.nmn3_modules import Modules
from models_vqa.nmn3_assembler import INVALID_EXPR
from models_vqa.question_prior_net import question_prior_net

from util.cnn import fc_layer as fc, conv_layer as conv

class NMN3Model:
    def __init__(self, image_feat_grid, text_seq_batch, seq_length_batch,
        T_decoder, num_vocab_txt, embed_dim_txt, num_vocab_nmn,
        embed_dim_nmn, lstm_dim, num_layers, assembler,
        encoder_dropout, decoder_dropout, decoder_sampling,
        num_choices, use_qpn, qpn_dropout, reduce_visfeat_dim=False, new_visfeat_dim=256,
        use_gt_layout=None, gt_layout_batch=None,
        scope='neural_module_network', reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            # Part 0: Visual feature from CNN
            self.reduce_visfeat_dim = reduce_visfeat_dim
            if reduce_visfeat_dim:
                # use an extrac linear 1x1 conv layer (without ReLU)
                # to reduce the feature dimension
                with tf.variable_scope('reduce_visfeat_dim'):
                    image_feat_grid = conv('conv_reduce_visfeat_dim', image_feat_grid, kernel_size=1, stride=1, output_dim=new_visfeat_dim)
                print('visual feature dimension reduced to %d' % new_visfeat_dim)
            self.image_feat_grid = image_feat_grid

            # Part 1: Seq2seq RNN to generate module layout tokens
            with tf.variable_scope('layout_generation'):
                att_seq2seq = AttentionSeq2Seq(text_seq_batch,                      #num_vocab_txt: 17742
                    seq_length_batch, T_decoder, num_vocab_txt,                     #T_decoder: 13
                    embed_dim_txt, num_vocab_nmn, embed_dim_nmn, lstm_dim,          #embed_dim_txt: 300
                    num_layers, assembler, encoder_dropout, decoder_dropout,        #num_vocab_nmn: 15
                    decoder_sampling, use_gt_layout, gt_layout_batch)               #encoder_dropout: True, decoder_dropout: True
                self.att_seq2seq = att_seq2seq                                      #embed_dim_nmn: 300, use_gt_layout: True
                                                                                    #lstm_dim: 1000
                predicted_tokens = att_seq2seq.predicted_tokens
                token_probs = att_seq2seq.token_probs                               #att_seq2seq.decodes_ta[1].stack()
                word_vecs = att_seq2seq.word_vecs                                   #tf.reduce_sum(atts*self.embedded_input_seq, axis=1)
                neg_entropy = att_seq2seq.neg_entropy

                self.atts = att_seq2seq.atts
                self.predicted_tokens = predicted_tokens
                self.token_probs = token_probs
                self.word_vecs = word_vecs
                self.neg_entropy = neg_entropy

                # log probability of each generated sequence
                self.log_seq_prob = tf.reduce_sum(tf.log(token_probs), axis=0)

            # Part 2: Neural Module Network
            with tf.variable_scope('layout_execution'):                             #image_feat_grid: (?, 14, 14, 256)
                modules = Modules(image_feat_grid, word_vecs, None, num_choices)    #num_choices: 3001
                self.modules = modules                                              #word_vecs: (?, ?, encoder_embed_dim=300)

                #=================================================================
                # Recursion of modules
                att_shape = image_feat_grid.get_shape().as_list()[1:-1] + [1]       #att_shape: [14, 14, 1]
                # Declare Forward Declaration of module recursion
                att_expr_decl = td.ForwardDeclaration(td.PyObjectType(),            #in type
                                                      td.TensorType(att_shape))     #out type

                #----------------------------------------------------------------------------------------------
                # _Scene
                #case_scene.input_type: PyObjectType()
                #case_scene.output_type: TensorType((14, 14, 1), 'float32')

                #Record to create a tuple, Map to create a sequence:

                #The whole point of Fold is to get your data into TensorFlow; the Function block lets you convert a
                # TITO (Tensors In, Tensors Out) function to a block:
                case_scene = td.Record([('time_idx', td.Scalar(dtype='int32')),
                                       ('batch_idx', td.Scalar(dtype='int32'))])
                case_scene = case_scene >> td.Function(modules.SceneModule)

                # _Find
                case_find = td.Record([('time_idx', td.Scalar(dtype='int32')),
                                       ('batch_idx', td.Scalar(dtype='int32'))])
                case_find = case_find >> td.Function(modules.FindModule)

                # _Filter
                case_filter = td.Record([('input_0', att_expr_decl()),
                                         ('time_idx', td.Scalar(dtype='int32')),
                                         ('batch_idx', td.Scalar(dtype='int32'))])
                case_filter = case_filter >> td.Function(modules.FilterModule)

                # _FindSameProperty
                case_find_same_property = td.Record([('input_0', att_expr_decl()),
                                                     ('time_idx', td.Scalar(dtype='int32')),
                                                     ('batch_idx', td.Scalar(dtype='int32'))])
                case_find_same_property = case_find_same_property >> td.Function(modules.FindSamePropertyModule)

                # _Transform
                case_transform = td.Record([('input_0', att_expr_decl()),
                                            ('time_idx', td.Scalar('int32')),
                                            ('batch_idx', td.Scalar('int32'))])
                case_transform = case_transform >> td.Function(modules.TransformModule)

                # _And
                case_and = td.Record([('input_0', att_expr_decl()),
                                      ('input_1', att_expr_decl()),
                                      ('time_idx', td.Scalar('int32')),
                                      ('batch_idx', td.Scalar('int32'))])
                case_and = case_and >> td.Function(modules.AndModule)

                # _Or
                case_or = td.Record([('input_0', att_expr_decl()),
                                     ('input_1', att_expr_decl()),
                                     ('time_idx', td.Scalar('int32')),
                                     ('batch_idx', td.Scalar('int32'))])
                case_or = case_or >> td.Function(modules.OrModule)

                # _Exist
                case_exist = td.Record([('input_0', att_expr_decl()),
                                        ('time_idx', td.Scalar('int32')),
                                        ('batch_idx', td.Scalar('int32'))])
                case_exist = case_exist >> td.Function(modules.ExistModule)

                # _Count
                case_count = td.Record([('input_0', att_expr_decl()),
                                        ('time_idx', td.Scalar('int32')),
                                        ('batch_idx', td.Scalar('int32'))])
                case_count = case_count >> td.Function(modules.CountModule)  #<---

                # _EqualNum
                case_equal_num = td.Record([('input_0', att_expr_decl()),
                                            ('input_1', att_expr_decl()),
                                            ('time_idx', td.Scalar('int32')),
                                            ('batch_idx', td.Scalar('int32'))])
                case_equal_num = case_equal_num >> td.Function(modules.EqualNumModule)

                # _MoreNum
                case_more_num = td.Record([('input_0', att_expr_decl()),
                                            ('input_1', att_expr_decl()),
                                            ('time_idx', td.Scalar('int32')),
                                            ('batch_idx', td.Scalar('int32'))])
                case_more_num = case_more_num >> td.Function(modules.MoreNumModule)

                # _LessNum
                case_less_num = td.Record([('input_0', att_expr_decl()),
                                            ('input_1', att_expr_decl()),
                                            ('time_idx', td.Scalar('int32')),
                                            ('batch_idx', td.Scalar('int32'))])
                case_less_num = case_less_num >> td.Function(modules.LessNumModule)
                # _SameProperty
                case_same_property = td.Record([('input_0', att_expr_decl()),
                                                ('input_1', att_expr_decl()),
                                                ('time_idx', td.Scalar('int32')),
                                                ('batch_idx', td.Scalar('int32'))])
                case_same_property = case_same_property >> td.Function(modules.SamePropertyModule)
                # _Describe
                case_describe = td.Record([('input_0', att_expr_decl()),
                                           ('time_idx', td.Scalar('int32')),
                                           ('batch_idx', td.Scalar('int32'))])
                case_describe = case_describe >> td.Function(modules.DescribeModule)

                # ============================================================================
                #recursion_cases:
                #   <td.OneOf>: choose of these children blocks as the beginning of recursion
                #               and assign it to 'recursion_cases' variable
                recursion_cases = td.OneOf(td.GetItem('module'), {  '_Scene': case_scene,
                                                                    '_Find': case_find,
                                                                    '_Filter': case_filter,
                                                                    '_FindSameProperty': case_find_same_property,
                                                                    '_Transform': case_transform,
                                                                    '_And': case_and,
                                                                    '_Or': case_or})
                #to specify that att_expr_decl() can be any of the above
                #resolve_to(): Resolve the forward declaration by setting it to the given block
                att_expr_decl.resolve_to(recursion_cases)
                # ============================================================================


                # ============================================================================
                # For invalid expressions, define a dummy answer
                # so that all answers have the same form
                dummy_scores = td.Void() >> td.FromTensor(np.zeros(num_choices, np.float32))

                #output_scores: <td.OneOf>
                output_scores = td.OneOf(td.GetItem('module'), {'_Exist': case_exist,
                                                                '_Count': case_count,           #<-- add expert to count
                                                                '_EqualNum': case_equal_num,
                                                                '_MoreNum': case_more_num,
                                                                '_LessNum': case_less_num,
                                                                '_SameProperty': case_same_property,
                                                                '_Describe': case_describe,
                                                                INVALID_EXPR: dummy_scores})

                #ipdb.set_trace()
                # compile and get the output scores
                #self.compiler.output_tensors
                #[<tf.Tensor 'neural_module_network/layout_execution/output_gathers/float32_3001__TensorFlowFoldOutputTag_0:0'
                # shape=(?, 3001) dtype=float32>]

                #Creates a Compiler, compiles a block, and initializes loom
                self.compiler = td.Compiler.create(output_scores)
                self.scores_nmn = self.compiler.output_tensors[0]

                self.att_expr_decl = att_expr_decl
                self.recursion_cases = recursion_cases
                self.output_scores = output_scores

                #self.compiler2 = td.Compiler.create(recursion_cases)
                #self.scores_nmn2 = self.compiler2.output_tensors[0]    #(?, 14, 14, 1)

                # self.compiler.output_tensors[0]:
                #   <tf.Tensor 'neural_module_network/layout_execution/output_gathers/float32_3001__TensorFlowFoldOutputTag_0:0' shape=(?, 3001) dtype=float32>
                # self.modules.image_feat_grid_with_coords
                #   <tf.Tensor 'neural_module_network/layout_execution/concat_1:0' shape=(?, 14, 14, 258) dtype=float32>
                # self.modules.coords_map
                #   < tf.Tensor 'neural_module_network/layout_execution/StopGradient:0' shape = (?, 14, 14, 2) dtype = float32 >
                # ============================================================================

            # Add a question prior network if specified
            self.use_qpn = use_qpn                              #True
            self.qpn_dropout = qpn_dropout                      #True
            if use_qpn:                                         #<-- needed? set to False
                #att_seq2seq.encoder_states[0]: (?, 1000)
                #att_seq2seq.encoder_states[1]: (?, 1000)
                self.scores_qpn = question_prior_net(att_seq2seq.encoder_states, num_choices, qpn_dropout)
                self.scores = self.scores_nmn + self.scores_qpn
            else:
                self.scores = self.scores_nmn

            # Regularization: Entropy + L2
            self.entropy_reg = tf.reduce_mean(neg_entropy)
            module_weights = [v for v in tf.trainable_variables()
                              if (scope in v.op.name and v.op.name.endswith('weights'))]
            self.l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in module_weights])

        #ipdb.set_trace()
        #print('.')

