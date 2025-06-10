import numpy as np
import matplotlib.pyplot as plt
import random

def get_domains(start, end, num_low, num_high):
    return np.linspace(start, end, num_low), np.linspace(start, end, num_high)

def plot_signals(
    low_res_signals,
    high_res_signals,
    model_signals=None,
    sharex=True,
    sharey=False,
    rows=3,
    cols=1,
    start=0,
    end=4*np.pi,
    show=True
):

    f, axes = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, figsize=(20, 10))
    x_low, x_high = get_domains(start, end, num_low=low_res_signals.size(0), num_high=high_res_signals.size(0))

    # print(axes)
    # for ax in axes:
    # index = random.randint(0, len(low_res_signals) - 1)
    y_true = high_res_signals
    y_low = low_res_signals

    axes.plot(x_high, y_true)
    axes.plot(x_low, y_low, "o")

    if model_signals is not None:
        y_pred = model_signals
        axes.plot(x_high, y_pred)

    f.legend(
        ["True Signal", "Sub sample points", "Approximation"],
        loc="upper center",
        ncols=3,
    )
    
    if(show):
        plt.show()
    
    plt.close()
    
    return f
