import matplotlib.pyplot as plt

def report(actual_labels, predicted_labels):
    classes = [0, 1]
    epsilon = 1e-6

    tps = {}
    tns = {}
    fps = {}
    fns = {}

    for c in classes:
        tps[c] = 0
        tns[c] = 0
        fps[c] = 0
        fns[c] = 0

    for actual, predicted in zip(actual_labels, predicted_labels):
        for c in classes:
            if c == actual and c == predicted:
                tps[c] += 1
            elif c == actual and c != predicted:
                fns[c] += 1
            elif c != actual and c == predicted:
                fps[c] += 1
            elif c != actual and c != predicted:
                tns[c] += 1

    precision = {}
    recall = {}
    f1 = {}

    for c in classes:
        precision[c] = tps[c] / (tps[c] + fps[c] + epsilon)
        recall[c] = tps[c] / (tps[c] + fns[c] + epsilon)
        f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c] + epsilon)

    accuracy = sum(tps.values()) / (sum(tps.values()) + sum(fps.values()) + sum(fns.values()))

    out = "Report\n"
    for c in classes:
        out += "Class {}\n".format(c)
        out += "Precision: {:.3f}\n".format(precision[c])
        out += "Recall: {:.3f}\n".format(recall[c])
        out += "F1-Score: {:.3f}\n\n".format(f1[c])
    out += "Accuracy: {:.3f}\n\n".format(accuracy)
    
    return out 

def accuracy(actual_labels, predicted_labels):
    classes = [0, 1]

    tps = {}
    tns = {}
    fps = {}
    fns = {}

    for c in classes:
        tps[c] = 0
        tns[c] = 0
        fps[c] = 0
        fns[c] = 0

    for actual, predicted in zip(actual_labels, predicted_labels):
        for c in classes:
            if c == actual and c == predicted:
                tps[c] += 1
            elif c == actual and c != predicted:
                fns[c] += 1
            elif c != actual and c == predicted:
                fps[c] += 1
            elif c != actual and c != predicted:
                tns[c] += 1

    return sum(tps.values()) / (sum(tps.values()) + sum(fps.values()) + sum(fns.values()))

def plot_graphs(results, title, cmap='Spectral'):
    fig = plt.figure() 
    fig.set_size_inches(18, 4)

    X = results['eps']
    Y = results['min_samples']

    ax = fig.add_subplot(1, 4, 1, projection='3d')
    Z = results['accuracy']
    ax.scatter(X, Y, Z, c=Z, cmap=cmap)
    ax.set_xlabel('Eps')
    ax.set_ylabel('MinPts')
    ax.set_zlabel('Accuracy')
    ax.set_title('Accuracy')

    ax = fig.add_subplot(1, 4, 2, projection='3d')
    Z = results['score']
    ax.scatter(X, Y, Z, c=Z, cmap=cmap)
    ax.set_xlabel('Eps')
    ax.set_ylabel('MinPts')
    ax.set_zlabel('Score')
    ax.set_title('Adjusted Rand Score')

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    Z = results['noise']
    ax.scatter(X, Y, Z, c=Z, cmap=cmap)
    ax.set_xlabel('Eps')
    ax.set_ylabel('MinPts')
    ax.set_zlabel('#Noise')
    ax.set_title('Number of noise points')

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    Z = results['nb_clusters']
    ax.scatter(X, Y, Z, c=Z, cmap=cmap)
    ax.set_xlabel('Eps')
    ax.set_ylabel('MinPts')
    ax.set_zlabel('#Clusters')
    ax.set_title('Number of Clusters')

    fig.text(0.5, 0.01, title, ha='center', fontsize=13)
    plt.show()
    