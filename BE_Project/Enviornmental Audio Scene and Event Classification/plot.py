import matplotlib.pyplot as plt

# PLOT AUDIO FILE
def plotaudio(output,label):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    plt.title(label)
    plt.plot(output,color= "blue")
    ax.set_xlim((0,len(output)))
    ax.margins(2,-0.1)
    plt.show(block = False)