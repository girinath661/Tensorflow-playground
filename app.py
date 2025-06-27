import streamlit as st
from tensorflow import keras
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from keras.activations import relu,sigmoid,tanh
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Nadam




st.title("Implementation of Tensorflow Playground")

dataset=st.selectbox('Choose a dataset type',
    options=['make_blobs', 'make_circles', 'make_moons'])

if dataset=='make_blobs':
    X, y = make_blobs(n_samples=1000,random_state=42,centers=2, cluster_std=1.5)
    
elif dataset =='make_circles':
    X, y = make_circles(n_samples=1000,random_state=42,noise=0.2,factor=0.5)
    
else:
    X, y = make_moons(n_samples=1000, random_state=42,noise=0.2)
    


training_Split_Ratio=st.number_input('Enter Train-Test Split Ratio',min_value=0.1,step=0.1)
batch=st.number_input('Enter Batch Size',min_value=8,step=8)
no_hidden_layer=st.number_input('Enter no.of hidden layers',min_value=1)

layers=[]
with st.expander("🔧 Configure Hidden Layers"):
    for i in range(1, no_hidden_layer + 1):
        st.markdown(f"**Hidden Layer {i}**")
        cols = st.columns(2)
        with cols[0]:
            neurons = st.number_input(f'Neurons in Layer {i}', min_value=5, step=1, key=f"neurons_{i}")
        with cols[1]:
            activation_fun = st.selectbox(f'Activation of Layer {i}', ['relu', 'sigmoid', 'tanh'], key=f"activation_{i}")
        layers.append((neurons, activation_fun))


optimizers = st.selectbox("Select Optimizer", ['adam', 'sgd', 'rmsprop', 'adagrad'])
learning_rate = st.number_input('Enter Learning Rate', min_value=0.001, max_value=1.0, step=0.001, format="%.4f")
epoch=st.number_input('Enter no.of epochs',min_value=15,step=1)

if optimizers == 'adam':
    optimizer = Adam(learning_rate=learning_rate)
elif optimizers == 'sgd':
    optimizer = SGD(learning_rate=learning_rate)
elif optimizers == 'rmsprop':
    optimizer = RMSprop(learning_rate=learning_rate)
elif optimizers == 'adagrad':
    optimizer = Adagrad(learning_rate=learning_rate)

if st.button('Train Model'):
    model=keras.Sequential()
    model.add(keras.layers.Input(shape=(X.shape[1],)))

    for neurons, activation in layers:
        model.add(keras.layers.Dense(neurons, activation=activation))
    if dataset=='make_blobs':
        model.add(keras.layers.Dense(3,activation="softmax"))
        model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=['accuracy'])
    else:
        model.add(keras.layers.Dense(1,activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])


    history=model.fit(X,y,validation_split=training_Split_Ratio,epochs=epoch,batch_size = batch,verbose=0)




    st.success("Model training complete!")
    
    st.subheader("Decision Boundary")

    def plot_decision_boundary(model, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                            np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = model.predict(grid)

        if probs.ndim > 1 and probs.shape[1] > 1:
            Z = np.argmax(probs, axis=1).reshape(xx.shape)
        else:
            Z = (probs > 0.5).astype(int).reshape(xx.shape)

        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, alpha=0.5, cmap="coolwarm")
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors='k')
        plt.xticks([]), plt.yticks([])
        st.pyplot(plt)
        plt.close()
    plot_decision_boundary(model, X, y)


    st.subheader("Training vs Validation Loss")
    fig2, ax2 = plt.subplots()
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Training vs Testing Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    
    st.subheader('Metrics')
    st.write("✅ Training Accuracy :", f"{history.history["accuracy"][-1]:.4f}")
    st.write("🤖 Testing Accuracy :", f"{history.history["val_accuracy"][-1]:.4f}")
    