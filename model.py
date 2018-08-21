import os
import sys
import numpy as np
import tensorflow as tf
import utils


class TranscriptNet:
    def __init__(self, config):
        self.config = config
        self.word2idx = None
        self.idx2word = None
        self.vocab_size = None

    def __build_model(self):

        with tf.variable_scope('TranscriptNet'):

            self.feed_dict = {}

            if self.config.mode in 'test':
                self.config.batch_size = 1

            input_text = tf.placeholder(tf.int32, [None, None], name='input')
            targets = tf.placeholder(tf.int32, [None, None], name='targets')
            keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

            if self.config.word2vec:
                embedding_ph = tf.placeholder(tf.float32, [self.vocab_size, self.config.embed_size], name='embeddings')
                embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.config.embed_size]), trainable=False)
                embedding_init = embedding.assign(embedding_ph)
                self.feed_dict['embeddings'] = embedding_ph
                self.feed_dict['embeddings_init'] = embedding_init
            else:
                embedding = tf.Variable(tf.random_uniform((self.vocab_size, self.config.embed_size), -1, 1))
                embedding = tf.nn.dropout(embedding, keep_prob=keep_prob, noise_shape=[self.vocab_size, 1])

            embed = tf.nn.embedding_lookup(embedding, input_text)

            lstm = tf.contrib.rnn.BasicLSTMCell(self.config.lstm_size)

            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

            cell = tf.contrib.rnn.MultiRNNCell([drop] * self.config.lstm_layers)

            initial_state = cell.zero_state(self.config.batch_size, tf.float32)
            initial_state_named = tf.identity(initial_state, name="initial_state")

            outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
            final_state_named = tf.identity(final_state, name="final_state")

            predictions = tf.contrib.layers.fully_connected(outputs, self.vocab_size, activation_fn=None)

            probabilities = tf.nn.softmax(predictions, name='probabilities')

            self.feed_dict['input_text'] = input_text
            self.feed_dict['targets'] = targets
            self.feed_dict['keep_prob'] = keep_prob
            self.feed_dict['initial_state'] = initial_state_named
            self.feed_dict['final_state'] = final_state_named
            self.feed_dict['predictions'] = predictions
            self.feed_dict['probabilities'] = probabilities

    def __build_decoder(self):
        """
        Builds lstm decoder graph
        :return:
        """

    def __get_batches(self, text_seq):
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
                batches[batch, 0, sequence] = text_seq[index:index + self.config.seq_length]
                batches[batch, 1, sequence] = text_seq[index + 1:index + self.config.seq_length + 1]

        return batches

    def train(self, text):
        """
        Builds the cost function and optimizer and trains the model on the training data
        :param text: text data to train the model on
        """
        current_epoch = 0
        current_step = 0
        if self.config.resume:
            if os.path.isfile(os.path.join(self.config.train_dir, 'save.npy')):
                current_epoch, current_step = np.load(os.path.join(self.config.train_dir, 'save.npy'))
            else:
                print("\nNo checkpoints, initializing training...\n")
                self.config.resume = False

        embeddings = None
        if self.config.word2vec:
            embeddings = utils.word2vec(text, self.config.embed_size)

        tokens, self.word2idx, self.idx2word = utils.build_vocab(text)
        vocab = list(self.word2idx)
        self.vocab_size = len(vocab)

        utils.save_data(self.word2idx, self.idx2word, embeddings)

        self.__build_model()

        with tf.variable_scope('Loss'):
            cost = tf.contrib.seq2seq.sequence_loss(self.feed_dict['predictions'],
                                                    self.feed_dict['targets'],
                                                    tf.ones([self.config.batch_size,
                                                             tf.shape(self.feed_dict['input_text'])[1]]))

        global_step = tf.Variable(current_step, name='global_step', dtype=tf.int64)

        with tf.variable_scope('Optimizer'):
            starter_learning_rate = self.config.learning_rate
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step, 100, 0.95, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)

        text_seq = [self.word2idx[word] for word in tokens]
        batches = self.__get_batches(text_seq)

        train_summary = tf.summary.scalar('train_loss', cost)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.config.log_dir, graph=sess.graph)
            sess.run(tf.global_variables_initializer())

            if self.config.word2vec:
                sess.run(self.feed_dict['embeddings_init'], feed_dict={self.feed_dict['embeddings']: embeddings})

            if self.config.resume:
                print("\nLoading previously trained model...")
                print(current_epoch, "out of", self.config.epochs, "epochs completed in previous run.")
                try:
                    ckpt_file = os.path.join(self.config.train_dir, "model.ckpt-" + str(current_step))
                    saver.restore(sess, ckpt_file)
                    print("\nResuming training...\n")
                except Exception as e:
                    print(e)
                    print("\nCheckpoint not found, initializing training.\n")
                    sys.exit(-1)

            for epoch in range(current_epoch, self.config.epochs):
                state = sess.run(self.feed_dict['initial_state'], {self.feed_dict['input_text']: batches[0][0]})

                for batch, (x, y) in enumerate(batches):
                    feed = {self.feed_dict['input_text']: x,
                            self.feed_dict['targets']: y,
                            self.feed_dict['initial_state']: state,
                            self.feed_dict['keep_prob']: 1.0 - self.config.dropout
                            }
                    run = [global_step, cost, self.feed_dict['final_state'], train_op, train_summary]
                    step, train_loss, state, _, train_summ = sess.run(run, feed)

                    if step % 10 == 0:
                        writer.add_summary(train_summ, step)
                        writer.flush()

                    print('{}: Step {} : Batch {}/{} : train_loss = {:.3f}'.format(
                        epoch,
                        step,
                        batch,
                        len(batches),
                        train_loss))

                    if step % self.config.checkpoint_step == 0 and step:
                        print("Saving Model...")
                        saver.save(sess, os.path.join(self.config.train_dir, "model.ckpt"), global_step=step)
                        np.save(os.path.join(self.config.train_dir, "save"), (epoch, step))

                np.random.shuffle(batches)

            # Save Model
            saver.save(sess, self.config.load_model)
            print('Model Trained and saved as {}'.format(self.config.load_model))

    def __pick_word(self, probabilities):
        """
        Pick the next word in the generated text
        :param probabilities: Probabilites of the next word
        :return: String of the predicted word
        """
        probabilities = np.log(probabilities) / self.config.temperature
        exp_probs = np.exp(probabilities)
        probabilities = exp_probs / np.sum(exp_probs)
        pick = np.random.choice(len(probabilities), p=probabilities)
        return self.idx2word[pick]

    def generate(self, data_path="./training_data.pkl"):
        """
        Generates a Rick and Morty transcript starting from some seed word
        :return: transcript in str format
        """
        self.word2idx, self.idx2word, embeddings = utils.load_data(data_path)
        vocab = list(self.word2idx)
        self.vocab_size = len(vocab)

        self.__build_model()

        prime_word = 'rick:'
        gen_length = self.config.seq_length
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Load saved model
            saver.restore(sess, self.config.load_model)

            if self.config.word2vec:
                sess.run(self.feed_dict['embeddings_init'], feed_dict={self.feed_dict['embeddings']: embeddings})

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

                pred_word = self.__pick_word(probabilities[0][dyn_seq_length - 1])

                gen_sentences.append(pred_word)

            # Remove tokens
            token_dict = utils.token_lookup()
            script = ' '.join(gen_sentences)
            for key, token in token_dict.items():
                script = script.replace(' ' + token.lower(), key)
            script = script.replace('\n ', '\n')
            script = script.replace('( ', '(')
            script = script.replace("' ", "'")
            print()
            print(script)
            return script
