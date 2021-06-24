import numpy as np


class Net():
    def __init__(self, input_size, training_examples):
        self.layers = list()
        self.layer_size = list()
        self.layer_size.append(input_size)
        self.m = training_examples
        self.train_costs = []
        self.val_pred = []
        self.train_pred = []
        self.val_costs = []
        np.random.seed(0)

    def add_layer(self, neurons):
        l = dict()
        l['w'] = np.random.randn(neurons, self.layer_size[-1])/np.sqrt(neurons/2)
        l['b'] = np.zeros((neurons, 1))
        self.layers.append(l)
        self.layer_size.append(neurons)

    def linear(self, X, w, b):
        z = np.dot(w, X) + b
        return z

    def non_linearity(self, X, w, b, activation='relu'):
        z = self.linear(X, w, b)
        if activation == 'relu':
            A = np.maximum(0, z)

        elif activation == 'sigmoid':
            A = 1 / (1 + np.exp(-z))

        return z, A

    def softmax(self, A):
        return np.exp(A)/np.sum(np.exp(A), axis=0)

    def forward_prop(self, X):

        for i in range(len(self.layers)-1):
            z, A = self.non_linearity(X, self.layers[i]['w'], self.layers[i]['b'])
            self.layers[i]['z'] = z
            self.layers[i]['A'] = A
            X = A
        zL = self.linear(A, self.layers[-1]['w'], self.layers[-1]['b'])

        AL = self.softmax(zL)
        self.layers[-1]['z'] = zL
        self.layers[-1]['A'] = AL
        return AL

    def cost(self, AL, Y):
        logp = np.log(AL)
        cost = -np.sum(Y * logp, axis=0)
        cost = np.mean(cost)
        #print(cost.shape)
        return cost

    def gradients(self, dA, layer, A_prev, activation='relu'):
        m = dA.shape[1]
        if activation == 'relu':
            dz = dA
            dz[layer['z'] < 0] = 0
        elif activation == 'sigmoid':
            s = 1 / (1 + np.exp(-layer['z']))
            dz = dA * s * (1 - s)

        dw = (1/m) * np.dot(dz, A_prev.T)
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        dA_prev = np.dot(layer['w'].T, dz)
        return dw, db, dA_prev

    def back_prop(self, AL, X, Y):
        dAL = AL - Y
        self.layers[-1]['dA'] = dAL
        dA = dAL
        for i in range(len(self.layers)-1, -1, -1):
            if i != 0:
                dw, db, dA_prev = self.gradients(dA, self.layers[i], self.layers[i-1]['A'])
                self.layers[i]['dw'] = dw
                self.layers[i]['db'] = db
                self.layers[i-1]['dA'] = dA_prev
                dA = dA_prev
            elif i == 0:
                dw, db, dA_prev = self.gradients(dA, self.layers[i], X)
                self.layers[i]['dw'] = dw
                self.layers[i]['db'] = db

    def tune_params(self, learning_rate, lr_decay=True, decay_rate=10, t=1):
        if lr_decay:
            lr_decay = learning_rate / (1 + t / decay_rate)
            for i in range(len(self.layers)):
                self.layers[i]['w'] = self.layers[i]['w'] - lr_decay*self.layers[i]['dw']
                self.layers[i]['b'] = self.layers[i]['b'] - lr_decay*self.layers[i]['db']
        else:
            for i in range(len(self.layers)):
                self.layers[i]['w'] = self.layers[i]['w'] - learning_rate*self.layers[i]['dw']
                self.layers[i]['b'] = self.layers[i]['b'] - learning_rate*self.layers[i]['db']

    def predict(self, X):
        AL = self.forward_prop(X)
        y_predict = np.zeros(AL.shape)
        y_max = AL.max(axis=0)
        y_predict[AL == y_max] = 1
        return y_predict

    def accuracy(self, X, Y, AL):
        y_true = np.squeeze(np.array(AL[Y, np.arange(AL.shape[1])] == 1, dtype=np.int32))
        y_true = np.mean(y_true)
        return y_true

    def layer_stats(self):
        layer_mean = []
        layer_std = []
        for layer in self.layers:
            act_mean = np.mean(layer['z'])
            act_std = np.std(layer['z'])
            layer_mean.append(act_mean)
            layer_std.append(act_std)
        return layer_mean, layer_std

    def fit(self, X, Y, epochs, lr, mini_batch=False, mini_batch_size=0, train_val_split=0.2,
            decay_rate=int(), decay=False):
        split = int(self.m * (1 - train_val_split))
        X_train = X[:, :split]
        Y_train = Y[:, :split]
        X_val = X[:, split:]
        Y_val = Y[:, split:]
        M = int(X_train.shape[1]/mini_batch_size) - 1
        if mini_batch:
            for i in range(1, epochs + 1):
                temp = 0
                for j in range(M):
                    start = mini_batch_size*M
                    stop = mini_batch_size*(M+1)
                    AL = self.forward_prop(X_train[:, start:stop])
                    self.back_prop(AL, X_train[:, start:stop], Y_train[:, start:stop])
                    self.tune_params(lr, lr_decay=decay, decay_rate=decay_rate, t=i)
                    AL = self.forward_prop(X_train[:, start:stop])
                    cost = self.cost(AL, Y_train[:, start:stop])

                prediction_val = self.predict(X_val)
                prediction_train = self.predict(X_train)
                val_cost = self.cost(prediction_val, Y_val)
                self.train_costs.append(cost)
                self.val_costs.append(val_cost)
                self.val_pred.append(prediction_val)
                self.train_pred.append(prediction_train)

                print('Cost ={}, Epoch = {}'.format(cost, i))

        else:
            cost = 0
            AL = self.forward_prop(X_train)
            for i in range(1, epochs+1):
                self.back_prop(AL, X_train, Y_train)
                self.tune_params(lr)
                AL = self.forward_prop(X_train)
                cost = self.cost(AL, Y_train)
                #if i % 10 == 0:
                print('Cost = {}, Epoch = {}'.format(cost, i))
