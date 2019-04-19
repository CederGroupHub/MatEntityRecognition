import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import pickle
import logging
import numpy as np
theano.theano_logger.setLevel(logging.ERROR)

from .utils import shared, set_values, get_name, evaluate
from .nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward
from .optimization import Optimization

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He, Ziqin (Shaun) Rong'
__email__ = 'tanjin_he@berkeley.edu, rongzq08@gmail.com'

# Modified based on the NER Tagger code from arXiv:1603.01360 [cs.CL]


class Model_train(object):
    """
    Network architecture.
    """

    def __init__(self, model_path=None, model_name=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.

        :param model_path: File path to reload the model
        """
        # Model location
        if not model_path:
            if model_name:
                self.model_path = os.path.join('models', model_name) 
            else:
                self.model_path = os.path.join('models', 'model_0')
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            self.parameters_path = os.path.join(self.model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(self.model_path, 'mappings.pkl')
        else:
            # reload model parameters
            self.model_path = model_path
            self.parameters_path = os.path.join(self.model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(self.model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = pickle.load(f)

            with open(self.mappings_path, 'rb') as f:
                mappings = pickle.load(f)
            self.id_to_word = mappings['id_to_word']
            self.id_to_tag = mappings['id_to_tag']

        self.components = {}

        self.f_train = None
        self.f_eval = None

    def save_mappings(self, id_to_word, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.

        :param id_to_word: mapping from a number (id) to a word in text
        :param id_to_tag: mapping from a number (id) to a tag of word
        """
        self.id_to_word = id_to_word
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_tag': self.id_to_tag,
            }
            pickle.dump(mappings, f)

    def add_component(self, param):
        """
        Add a new parameter to the network.

        :param param: a dict of parameter names and parameter values
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in list(self.components.items()):
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in list(self.components.items()):
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])

    def build(self,
              dropout,
              input_vector,
              input_matrix,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              pre_emb,
              crf,
              training=True,
              **kwargs
              ):
        """
        Build the network.

        :param dropout: droupout rate
        :param char_dim: dimension of character feature
        :param char_lstm_dim: dimension of hidden layer for lstm dealing with character feature
        :param char_bidirect: use bidirectional lstm for character feature or not
        :param word_dim: dimension of word feature
        :param word_lstm_dim: dimension of hidden layer for lstm dealing with word embedding
        :param word_bidirect: use bidirectional lstm for word recognition or not
        :param lr_method: learning method
        :param pre_emb: pretrained embedding
        :param crf: use crf or not
        :param cap_dim: use capital character feature or not
        :param keyword_dim: dimension of keyword feature
        :param training: training or not
        :param kwargs: customized parameters of model
        :return f_train: training function
        :return f_eval: evaluation function
        """
        # save parameters
        if training:
            saved_locals = locals()
            self.parameters = saved_locals
            self.parameters.update(self.parameters['kwargs'])
            del self.parameters['kwargs']
            del self.parameters['self']
            # not save pre_emb because it might be very large 
            # and duplicates the original embedding file
            del self.parameters['pre_emb']
            del self.parameters['training']
            with open(self.parameters_path, 'wb') as f:
                pickle.dump(self.parameters, f)        

        # Training parameters
        n_words = len(self.id_to_word)
        n_tags = len(self.id_to_tag)

        # Network variables
        is_train = T.iscalar('is_train')
        tag_ids = T.ivector(name='tag_ids')

        if input_matrix:
            embedded_words = T.fmatrix(name='embedded_words')
        if input_vector and word_dim:
            word_ids = T.ivector(name='word_ids')

        # Sentence length
        s_len = (word_ids if input_vector else embedded_words).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        #
        # if input vector, use embedding to map word_ids
        #
        if input_vector:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            inputs.append(word_input)
            # Initialize with pretrained embeddings
            if type(pre_emb) == np.ndarray:
                word_layer.embeddings.set_value(pre_emb)

        #
        # if input matrix, use the input directly
        #
        if input_matrix:
            input_dim += embedded_words.shape[1]
            inputs.append(embedded_words)


        # Prepare final input
        inputs = T.concatenate(inputs, axis=1)
        
        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev')
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]
        if word_bidirect:
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=1
            )
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_for_output

        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')
            small = -1000
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
            observations = T.concatenate(
                [tags_scores, small * T.ones((s_len, 2))],
                axis=1
            )
            observations = T.concatenate(
                [b_s, observations, e_s],
                axis=0
            )

            # Score from tags
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

            # Score from transitions
            b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
            e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
            padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
            real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()

            all_paths_scores = forward(observations, transitions)
            cost = - (real_path_score - all_paths_scores)

        # Network parameters
        params = []
        if input_vector and (type(pre_emb) != np.ndarray):
            self.add_component(word_layer)
            params.extend(word_layer.params)
        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)
        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)

        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(transitions)
            params.append(transitions)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)

        # Prepare train and eval inputs
        eval_inputs = []
        if input_vector:
            eval_inputs.append(word_ids)
        if input_matrix:
            eval_inputs.append(embedded_words)
        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        # print('Compiling...')
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=forward(observations, transitions, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )

        self.f_train = f_train
        self.f_eval = f_eval 
        return f_train, f_eval


    def fit(self, input_X, input_Y, n_epochs=100, freq_eval=1000,
            dev_X=None, dev_Y=None, dev_sentences=None, 
            test_X=None, test_Y=None, test_sentences=None):
        """
        Train network.

        :param input_X: 2d array if using embedding, 3d array is not using embedding
        :param input_Y: 2d array of tag ids of words in sentences
        :param n_epochs: number of epochs
        :param freq_eval: frequency to evaluate with validation/test sets
        :param dev_X: input_X for validation set
        :param dev_Y: input_Y for validation set
        :param dev_sentences: original sentences in validation set
        :param test_X: input_X for test set
        :param test_Y: input_Y for test set
        :param dev_sentences: original sentences in validation set
        """

        best_dev = -np.inf
        best_test = -np.inf
        info_report = {}
        count = 0

        # assert same number of sentences
        assert len(input_X) == len(input_Y)

        train_data = list(zip(input_X, input_Y))

        for epoch in range(n_epochs):
            epoch_costs = []
            print("Starting epoch %i..." % epoch)
            for i, index in enumerate(np.random.permutation(len(train_data))):
                count += 1

                input = train_data[index]
                new_cost = self.f_train(*input)
                epoch_costs.append(new_cost)
                if i % 50 == 0 and i > 0 == 0:
                    print("%i, cost average: %f" % (i, np.mean(epoch_costs[-50:])))
                if count % freq_eval == 0 and (dev_X and test_X):
                    dev_Y_pred = self.predict(dev_X)
                    dev_score = evaluate(dev_Y_pred, dev_Y, dev_sentences, 
                                        self.id_to_tag, self.parameters['tag_scheme'])
                    test_Y_pred = self.predict(test_X)
                    test_score = evaluate(test_Y_pred, test_Y, test_sentences, 
                                        self.id_to_tag, self.parameters['tag_scheme'])
                    print("Score on dev: %.5f" % dev_score)
                    print("Score on test: %.5f" % test_score)
                    if dev_score > best_dev:
                        best_dev = dev_score
                        info_report = {'epoch with best dev score': epoch,
                                        'best dev score': best_dev,
                                        'test score in this epoch': test_score,
                                     }
                        print("New best score on dev.")
                        print("Saving model to disk...")
                        self.save()
                    if test_score > best_test:
                        best_test = test_score
                        print("New best score on test.")
                elif epoch == n_epochs - 1 and (not (dev_X and test_X)):
                    self.save()
            print("Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs)))

        return info_report


    def predict(self, input_X):
        """
        Predict with trained model.

        :param input_X: 2d array if using embedding, 3d array is not using embedding
        :return Y_predictions: 2d array of tag ids of words in sentences
        """
        # goal
        Y_predictions = []

        for i in range(len(input_X)):
            # Prediction
            if self.parameters['crf']:
                Y_pred = np.array(self.f_eval(input_X[i]))[1:-1]
            else:
                Y_pred = self.f_eval(*input_X[i]).argmax(axis=1)
            Y_predictions.append(Y_pred)

        return Y_predictions


    def predict_label(self, input_X):
        """
        Predict with trained model.

        :param input_X: 2d array if using embedding, 3d array is not using embedding
        :return label_predictions: 2d array of tag names of words in sentences
        """
        # goal
        label_predictions = []

        Y_predictions = self.predict(input_X)
        for i in range(len(Y_predictions)):
            Y_pred = [self.id_to_tag[y] for y in Y_predictions[i]]
            label_predictions.append(Y_pred)
        return label_predictions

