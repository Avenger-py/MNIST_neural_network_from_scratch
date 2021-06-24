import numpy as np
from mlp import Net
from matplotlib import pyplot as plt

train_data = np.load('train.npy')

X = train_data[:, 1:].reshape((60000, 784))
X = X.T.reshape(784, 60000)
Y = np.array(train_data[:, 0], dtype=np.int32)

X = X/255.0
X_mean = np.mean(X, axis=1).reshape((784, 1))
X = X - X_mean


def one_hot(y, classes, examples):
    #output y of shape (classes, examples)
    y = np.array(y, dtype=np.int32)
    y = y.reshape((-1, examples))
    n = np.array([i for i in range(classes)])
    y_onehot = np.zeros((n.size, examples))
    y_onehot[y[0], np.arange(y.size)] = 1
    return y_onehot


Ytrain = one_hot(Y, 10, 60000)


LR = 0.008
BATCH_SIZE = 1024
EPOCHS = 100
TRAIN_VAL_SPLIT = 0.2
DECAY_RATE = 20
split = int(X.shape[1] * (1-TRAIN_VAL_SPLIT))


model = Net(784, X.shape[1])
model.add_layer(512)
model.add_layer(10)
model.fit(X, Ytrain, epochs=EPOCHS, lr=LR, mini_batch=True, mini_batch_size=BATCH_SIZE,
          train_val_split=TRAIN_VAL_SPLIT, decay_rate=DECAY_RATE, decay=False)

Y = Y.reshape((1, -1))

val_preds = model.val_pred
val_accuracy = []
for i in val_preds:
    val_acc = model.accuracy(X[:, split:], Y[:, split:], i)
    val_accuracy.append(val_acc)

train_preds = model.train_pred
train_accuracy = []
for i in train_preds:
    train_acc = model.accuracy(X[:, :split], Y[:, :split], i)
    train_accuracy.append(train_acc)

train_costs = model.train_costs


plt.figure()
plt.subplot(121)
plt.plot([i for i in range(len(train_costs))], train_costs)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(['train cost'])

plt.subplot(122)
plt.plot([i for i in range(len(val_accuracy))], val_accuracy)
plt.plot([i for i in range(len(train_accuracy))], train_accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['val accuracy', 'train accuracy'])
plt.show()

print('Training accuracy = {}'.format(train_accuracy[-1]))
print('Validation accuracy = {}'.format(val_accuracy[-1]))
