import numpy as np
import tensorflow as tf
import utils


class TranscriptNet:
    def __init__(self, config):
        self.config = config
        self.word2idx, self.idx2word = utils.load_data('preprocessed_data.pkl')
        self.vocab_size = len(self.word2idx)
        self._build_model()

    def _build_model(self):

        with tf.variable_scope('TranscriptNet'):

            input_text = tf.placeholder(tf.int32, [None, None], name='input')
            targets = tf.placeholder(tf.int32, [None, None], name='targets')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

            embedding = tf.Variable(tf.random_uniform((self.vocab_size, self.config.embed_size), -1, 1))
            embed = tf.nn.embedding_lookup(embedding, input_text)

            lstm = tf.contrib.rnn.BasicLSTMCell(self.config.lstm_size)

            # drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

            cell = tf.contrib.rnn.MultiRNNCell([lstm] * self.config.lstm_layers)

            initial_state = cell.zero_state(self.config.batch_size, tf.float32)
            initial_state_named = tf.identity(initial_state, name="initial_state")

            outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
            final_state_named = tf.identity(final_state, name="final_state")

            predictions = tf.contrib.layers.fully_connected(outputs, self.vocab_size, activation_fn=None)

            probabilities = tf.nn.softmax(predictions, name='probabilities')

            self.feed_dict = {'input_text': input_text,
                              'targets': targets,
                              'learning_rate': learning_rate,
                              'keep_prob': keep_prob,
                              'initial_state': initial_state_named,
                              'final_state': final_state_named,
                              'probabilities': probabilities
                              }

    def get_batches(self, text_seq):
        """
        Return batches of input and target
        :param text_seq: Text with the words replaced by their ids
        :return: Batches as a Numpy array
        """
        num_batches = len(text_seq) // (self.config.batch_size * self.config.seq_length)
        block_size = num_batches * self.config.seq_length
        text_seq[num_batches * self.config.batch_size * self.config.seq_length] = text_seq[0]
        batches = np.zeros([num_batches, 2, self.config.batch_size, self.config.seq_length], dtype=np.int32)

        for batch in range(0, num_batches):
            for sequence in range(0, self.config.batch_size):
                index = batch * self.config.seq_length + sequence * block_size
                batches[batch, 0, sequence,] = text_seq[index:index + self.config.seq_length]
                batches[batch, 1, sequence,] = text_seq[index + 1:index + self.config.seq_length + 1]

        return batches

    def train(self, text_seq):
        """
        Builds the cost function and optimizer and trains the model on the training data
        """

        cost = tf.contrib.seq2seq.sequence_loss(self.feed_dict['probabilities'],
                                                self.feed_dict['targets'],
                                                tf.ones([self.config.batch_size,
                                                         tf.shape(self.feed_dict['input_text'])[1]]))

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

        batches = self.get_batches(text_seq)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.config.epochs):
                state = sess.run(self.feed_dict['initial_state'], {self.feed_dict['input_text']: batches[0][0]})

                for batch, (x, y) in enumerate(batches):
                    feed = {self.feed_dict['input_text']: x,
                            self.feed_dict['targets']: y,
                            self.feed_dict['initial_state']: state,
                            self.feed_dict['learning_rate']: self.config.learning_rate,
                            self.feed_dict['keep_prob']: 0.3
                            }
                    train_loss, state, _ = sess.run([cost, self.feed_dict['final_state'], train_op], feed)

                    # if (epoch * len(batches) + batch) % 10 == 0:
                    print('Epoch {}: Step {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch,
                        batch,
                        len(batches),
                        train_loss))

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, self.config.train_dir)
            print('Model Trained and Saved')

    def get_tensors(self, loaded_graph):
        """
        Get input, initial state, final state, and probabilities tensor from <loaded_graph>
        :param loaded_graph: TensorFlow graph loaded from file
        :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
        """
        input_tensor = loaded_graph.get_tensor_by_name("TranscriptNet/input:0")
        initial_state_tensor = loaded_graph.get_tensor_by_name("TranscriptNet/initial_state:0")
        final_state_tensor = loaded_graph.get_tensor_by_name("TranscriptNet/final_state:0")
        probs_tensor = loaded_graph.get_tensor_by_name("TranscriptNet/probabilities:0")
        return input_tensor, initial_state_tensor, final_state_tensor, probs_tensor

    def pick_word(self, probabilities):
        """
        Pick the next word in the generated text
        :param probabilities: Probabilites of the next word
        :param int_to_vocab: Dictionary of word ids as the keys and words as the values
        :return: String of the predicted word
        """

        pick = np.random.choice(len(probabilities), p=probabilities)

        return self.idx2word[pick]

    def generate(self):
        prime_word = 'rick'
        gen_length = 20
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Load saved model
            saver.restore(sess, self.config.load_model)
            # loader = tf.train.import_meta_graph('./train/model.ckpt.meta')
            # loader.restore(sess, './train/model.ckpt')

            # Sentences generation setup
            gen_sentences = [prime_word]
            prev_state = sess.run(self.feed_dict['initial_state'], {self.feed_dict['input_text']: np.array([[1]])})

            # Generate sentences
            for n in range(gen_length):
                # Dynamic Input
                dyn_input = [[self.word2idx[word] for word in gen_sentences[-self.config.seq_length:]]]
                dyn_seq_length = len(dyn_input[0])

                # Get Prediction
                probabilities, prev_state = sess.run(
                    [self.feed_dict['probabilities'], self.feed_dict['final_state']],
                    {self.feed_dict['input_text']: dyn_input, self.feed_dict['initial_state']: prev_state})

                pred_word = self.pick_word(probabilities[0][dyn_seq_length - 1])

                gen_sentences.append(pred_word)

            # Remove tokens
            token_dict = {';': "<semicolon>",
                          ':': "<colon>",
                          "'": "<inverted_comma>",
                          '"': "<quotation_mark>",
                          ',': "<comma>",
                          '\n': "<new_line>",
                          '!': "<exclamation_mark>",
                          '-': "<hyphen>",
                          '--': "<hyphens>",
                          '.': "<period>",
                          '?': "<question_mark>",
                          '(': "<left_paren>",
                          ')': "<right_paren>",
                          '♪': "<music_note>",
                          '[': "<left_square>",
                          ']': "<right_square>",
                          }
            tv_script = ' '.join(gen_sentences)
            for key, token in token_dict.items():
                ending = ' ' if key in ['\n', '(', '"'] else ''
                tv_script = tv_script.replace(' ' + token.lower(), key)
            tv_script = tv_script.replace('\n ', '\n')
            tv_script = tv_script.replace('( ', '(')
            print()
            print(tv_script)