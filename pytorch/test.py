from torchvision import datasets

path2data = './data'


train_data = datasets.MNIST(path2data, train=True, download=True)
x_train, y_train = train_data.data, train_data.targets

print(x_train[:10])
print(y_train[:10])