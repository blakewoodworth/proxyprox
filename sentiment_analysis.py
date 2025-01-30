import jax
import jax.numpy as jnp
import scipy as sp
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from proxy_alg import *
from data import *


def sentiment_analysis(proxy):
    objective_args = {
        'batch_size': 200
    }

    amazon_data = AmazonSentiment(n_examples=50000)
    target = 'Amazon'
    target_loss = LogisticRegression(amazon_data.target, amazon_data.test, objective_args)

    if proxy == 'Yelp':
        yelp_data = YelpSentiment(n_examples=50000, proxy_label_1=True, vectorizer=amazon_data.vectorizer)
        proxy_loss = LogisticRegression(yelp_data.proxy, amazon_data.test, objective_args)
    elif proxy == 'Shakespeare':
        shakespeare_data = ShakespeareText(n_examples=50000, vectorizer=amazon_data.vectorizer)
        proxy_loss = LogisticRegression(shakespeare_data.proxy, amazon_data.test, objective_args)
    else:
        print(f'Proxy {proxy} is not supported')
        return

    n_outer = 2*(target_loss.train_n // objective_args['batch_size'])
    optimization_args = {
        'n_outer': n_outer,
        'n_inner': 32,
        'eta': 0.1,
        'lr': 0.13
    }

    optimizer = ProxyProx(target_loss, proxy_loss, optimization_args)


    print('Target loss:', target)
    print('Proxy loss: ', proxy)

    ###############################################################


    ## tune SGD baseline lr
    optimizer.args['n_inner'] = 1

    infos = []
    loss_curves = []
    acc_curves = []

    lrs = np.power(10, np.linspace(0., 2., 25))
    for lr in tqdm(lrs):
        optimizer.args['lr'] = lr
        losses, accuracies = optimizer.train()
        if losses is None:
            continue
        else:
            infos.append(lr)
            loss_curves.append(losses)
            acc_curves.append(accuracies)

    best = []
    for i in range(len(infos)):
        val = np.mean(loss_curves[i][-5:])
        acc = np.mean(acc_curves[i][-5:])
        best.append((val, acc, infos[i], loss_curves[i], acc_curves[i]))

    best.sort(key=lambda x: x[0])
    best_sgd_lr = best[0][2]
    best_sgd_loss = best[0][3]
    best_sgd_acc = best[0][4]
    # print('Best SGD lr:', best_sgd_lr)

    optimizer.args['n_inner'] = 16
    infos = []
    loss_curves = []
    acc_curves = []

    lrs = np.power(10, np.linspace(0., 2., 20))
    etas = np.power(10, np.linspace(0., 3., 20))
    progress_bar = tqdm(total=len(lrs)*len(etas))
    for lr in lrs:
        for eta in etas:
            progress_bar.update()
            optimizer.args['lr'] = lr
            optimizer.args['eta'] = eta
            losses, accuracies = optimizer.train()
            if losses is None:
                continue
            else:
                infos.append((lr, eta))
                loss_curves.append(np.array(losses))
                acc_curves.append(np.array(accuracies))

    best = []
    for i in range(len(infos)):
        val = np.mean(loss_curves[i][-5:])
        acc = np.mean(acc_curves[i][-5:])
        best.append((val, acc, infos[i][0], infos[i][1], loss_curves[i], acc_curves[i]))

    n_best = 5
    best.sort(key=lambda x: x[0])
    lowest_loss = best[:n_best]
    # best.sort(key=lambda x: x[1], reverse=True)
    # highest_acc = best[:n_best]

    plt.figure()
    plt.title(f'Target: {target}, Proxy: {proxy}')
    plt.ylabel('Loss')
    plt.plot(best_sgd_loss, label='SGD baseline')
    for b in lowest_loss:
        _, _, lr, eta, losses, accs = b
        plt.plot(losses, label=f'lr={lr:.2f}, eta={eta:.2f}')
    plt.legend()
    plt.savefig(f'plots/loss-{target}_{proxy}_2.pdf')

    plt.figure()
    plt.title(f'Target: {target}, Proxy: {proxy}')
    plt.ylabel('Accuracy')
    plt.plot(best_sgd_acc, label='SGD baseline')
    for b in lowest_loss:
        _, _, lr, eta, losses, accs = b
        plt.plot(accs, label=f'lr={lr:.2f}, eta={eta:.2f}')
    plt.legend()
    plt.savefig(f'plots/acc-{target}_{proxy}_2.pdf')
    




