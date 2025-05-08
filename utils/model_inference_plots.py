import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from datetime import datetime, timezone


def plot_true_vs_pred(df, start_idx=0, end_idx=12500, dpi=100):
    df = df.loc[start_idx:end_idx, :].copy().reset_index(drop=True)
    df['Epoch_time'] = pd.to_datetime(df['Epoch_time'], unit='s', utc=True)
    last_idx = len(df) - 1

    plt.figure(figsize=(15, 6), dpi=dpi)

    plt.plot(df['Epoch_time'], df["Event_label_80"], label="True Event", color='black')

    in_event = False
    start = None
    label_added = False

    for i in range(len(df["Event_pred"])):
        if df["Event_pred"][i] == 1 and not in_event:
            in_event = True
            start = i
        elif df["Event_pred"][i] == 0 and in_event:
            in_event = False
            end = i - 1
            label = "Predicted Event" if not label_added else None
            plt.axvspan(df['Epoch_time'][start], df['Epoch_time'][end], ymin=0.045, ymax=0.955,
                           color='C0', alpha=0.3, label=label)
            label_added = True

    if in_event:
        label = "Predicted Event" if not label_added else None
        plt.axvspan(df['Epoch_time'][start], df['Epoch_time'][last_idx], ymin=0.045, ymax=0.955,
                       color='C0', alpha=0.3, label=label)
    
    plt.title("Predicted Events vs True Events on July 20th, 2013", fontsize=16)
    plt.ylabel("Event Indicator", labelpad=-15, fontsize=14)
    plt.yticks([0, 1], labels=['No Event', 'Event'], fontsize=12)
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=11))
    plt.xticks(fontsize=12)
    plt.xlabel("Time", fontsize=14)
    plt.show()


def plot_true_vs_pred_all(df, event_df, start_idx=0, end_idx=12500, dpi=100):
    df = df.loc[start_idx:end_idx, :].copy().reset_index(drop=True)
    df['Epoch_time'] = pd.to_datetime(df['Epoch_time'], unit='s', utc=True)
    last_idx = len(df) - 1

    fig, axs = plt.subplots(5, 1, figsize=(15, 25), dpi=dpi, sharex=True)

    axs[0].plot(df['Epoch_time'], df["Event_label_80"], label="True Event", color='black')

    in_event = False
    start = None
    label_added = False

    for i in range(len(df["Event_pred"])):
        if df["Event_pred"][i] == 1 and not in_event:
            in_event = True
            start = i
        elif df["Event_pred"][i] == 0 and in_event:
            in_event = False
            end = i - 1
            label = "Predicted Event" if not label_added else None
            axs[0].axvspan(df['Epoch_time'][start], df['Epoch_time'][end], ymin=0.045, ymax=0.955,
                           color='C0', alpha=0.3, label=label)
            label_added = True

    if in_event:
        label = "Predicted Event" if not label_added else None
        axs[0].axvspan(df['Epoch_time'][start], df['Epoch_time'][last_idx], ymin=0.045, ymax=0.955,
                       color='C0', alpha=0.3, label=label)
    
    event_ids = []
    event_ids = list(event_df[(pd.to_datetime(event_df['start_time']) >= df['Epoch_time'][0]) &
                              (pd.to_datetime(event_df['end_time']) <= df['Epoch_time'][last_idx])]
                              ['event_id'])
    
    axs[0].set_title(f"Predicted Events vs True Events (Event IDs: {', '.join(map(str, event_ids))})", fontsize=14)
    axs[0].set_ylabel("Event Indicator", labelpad=-15, fontsize=14)
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels(['No Event', 'Event'])
    axs[0].legend()

    axs[1].plot(df['Epoch_time'], df["Bx"], label='Bx', color='blue')
    axs[1].plot(df['Epoch_time'], df["By"], label='By', color='limegreen')
    axs[1].plot(df['Epoch_time'], df["Bz"], label='Bz', color='red')
    axs[1].set_title("Magnetic Field Components", fontsize=14)
    axs[1].set_ylabel("nT", fontsize=14)
    axs[1].legend(loc='upper right')

    axs[2].plot(df['Epoch_time'], df["Bx_rolling_stdev"], label='Bx Rolling Standard Deviation', color='blue')
    axs[2].plot(df['Epoch_time'], df["By_rolling_stdev"], label='By Rolling Standard Deviation', color='limegreen')
    axs[2].plot(df['Epoch_time'], df["Bz_rolling_stdev"], label='Bz Rolling Standard Deviation', color='red')
    axs[2].set_title("Magnetic Field 9-Window Rolling Standard Deviation", fontsize=14)
    axs[2].set_ylabel("Standard Deviation", fontsize=14)
    axs[2].legend(loc='upper right')

    axs[3].plot(df['Epoch_time'], df["Bx_conditional_vol"], label='Bx Volatility', color='blue')
    axs[3].plot(df['Epoch_time'], df["By_conditional_vol"], label='By Volatility', color='limegreen')
    axs[3].plot(df['Epoch_time'], df["Bz_conditional_vol"], label='Bz Volatility', color='red')
    axs[3].set_title("Magnetic Field Volatilities", fontsize=14)
    axs[3].set_ylabel("Volatility", fontsize=14)
    axs[3].legend(loc='upper right')

    axs[4].plot(df['Epoch_time'], df["|V_perp|"], label='Velocity', color='black')
    axs[4].set_title("Velocity", fontsize=14)
    axs[4].set_ylabel("Velocity", fontsize=14)
    axs[4].set_xlabel("Time", fontsize=14)

    axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    axs[4].xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xticks(rotation=45)
    plt.show()


def plot_true_vs_pred_probas(y_test, y_pred_probas, thershold=0.5, start_idx=0, end_idx=20000, dpi=100):
    plt.figure(figsize=(15, 6), dpi=dpi)
    plt.axhline(thershold, label='Probability Threshold', color='red', linestyle='--', alpha=0.4)
    plt.plot(y_test[start_idx:end_idx], label="True Event", color='black')
    plt.plot(y_pred_probas[start_idx:end_idx], label="Predicted Probability of Event", alpha=0.7)
    plt.title("Predicted Probability of Event vs True Event")
    plt.xlabel(f"Time ({end_idx - start_idx} Indices in Test Set)")
    plt.ylabel("Probability of Event")
    plt.yticks(np.arange(1.1, step=0.1))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.show()


def plot_positive_class_roc(y_true, y_score, pos_label=1, dpi=100):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6.5, 6), dpi=dpi)
    plt.plot(fpr, tpr, color='C0', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Class 1 (Event)', fontsize=16)
    plt.legend(loc="lower right")
    plt.show()


def plot_raw_confusion_matrix(y_test, y_pred, dpi=100):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=dpi)

    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax1)

    fmt = ',d'
    thresh = cm.max() / 2

    for i, j in np.ndindex(cm.shape):
        ax1.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=11)

    ax1.set_title('Confusion Matrix - Raw Values', fontsize=16)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xticks(np.arange(len(np.unique(y_test))))
    ax1.set_yticks(np.arange(len(np.unique(y_test))))
    ax1.set_xticklabels(['No Event', 'Event'])
    ax1.set_yticklabels(['No Event', 'Event'])
    plt.show()


def event_analysis(event_groups, ratio_threshold, side='lower', target='Event_label_80', predictions='Event_pred'):
    ratios_capture = []

    for (key, _,cls), group in event_groups:
        true_label = group[target].values
        pred_label = group[predictions].values
        
        N_j = np.sum(true_label)
        Nt_j = np.sum(pred_label == 1)

        ratio = Nt_j / N_j
        
        if side == "lower":
            if ratio < ratio_threshold:
                ratios_capture.append(group.index.tolist())
        elif side == "upper":
            if ratio >= ratio_threshold:
                ratios_capture.append(group.index.tolist())
    
    return ratios_capture


def extract_max_features(sequences, df, event_type=True):
    max_vels, max_rstdsx, max_rstdsy, max_rstdsz = [], [], [], []
    max_volsx, max_volsy, max_volsz = [], [], []

    for seq in sequences:
        if event_type or len(seq) > 150:
            cont_df = df.loc[seq[0]:seq[-1]].copy().reset_index(drop=True)
            max_vels.append(np.max(cont_df['|V_perp|']))
            max_rstdsx.append(np.max(cont_df['Bx_rolling_stdev']))
            max_rstdsy.append(np.max(cont_df['By_rolling_stdev']))
            max_rstdsz.append(np.max(cont_df['Bz_rolling_stdev']))
            max_volsx.append(np.max(cont_df['Bx_conditional_vol']))
            max_volsy.append(np.max(cont_df['By_conditional_vol']))
            max_volsz.append(np.max(cont_df['Bz_conditional_vol']))

    return {
        "velocity": max_vels,
        "Bx_stdev": max_rstdsx,
        "By_stdev": max_rstdsy,
        "Bz_stdev": max_rstdsz,
        "Bx_vol": max_volsx,
        "By_vol": max_volsy,
        "Bz_vol": max_volsz
    }


def plot_vol_stdev_histogram(data, event_type='Missed'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    bins = np.arange(0, 1.1, step=0.1)

    axes[0, 0].hist(data["Bx_vol"], color='C0', edgecolor="black", bins=bins)
    axes[0, 0].set_title(f"Max Volatility in Bx for {event_type} Events")

    axes[0, 1].hist(data["By_vol"], color='C2', edgecolor="black", bins=bins)
    axes[0, 1].set_title(f"Max Volatility in By for {event_type} Events")

    axes[0, 2].hist(data["Bz_vol"], color='C3', edgecolor="black", bins=bins)
    axes[0, 2].set_title(f"Max Volatility in Bz for {event_type} Events")

    axes[1, 0].hist(data["Bx_stdev"], color='C0', edgecolor="black", bins=bins)
    axes[1, 0].set_title(f"Max Rolling Stdev in Bx for {event_type} Events")

    axes[1, 1].hist(data["By_stdev"], color='C2', edgecolor="black", bins=bins)
    axes[1, 1].set_title(f"Max Rolling Stdev in By for {event_type} Events")

    axes[1, 2].hist(data["Bz_stdev"], color='C3', edgecolor="black", bins=bins)
    axes[1, 2].set_title(f"Max Rolling Stdev in Bz for {event_type} Events")

    plt.show()


def plot_vol_stdev_2d_histogram(data, event_type='Missed'):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    bins = np.arange(0.0, 1.05, step=0.05)
    components = ['Bx', 'By', 'Bz']
    for ax, comp in zip(axes, components):
        h = ax.hist2d(
            data[f'{comp}_stdev'],
            data[f'{comp}_vol'],
            bins=[bins, bins],
            cmap='plasma',
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_xlabel(f'{comp} Rolling Stdev')
        ax.set_ylabel(f'{comp} Conditional Volatility')
        ax.set_title(f'{comp} - {event_type} Events')

    fig.colorbar(h[3], ax=axes.ravel().tolist(), label='Counts')

    plt.show()


def plot_vol_stdev_2d_avg_histogram(data, event_type='Missed'):
    plt.figure(figsize=(6, 4.5))
    bins = np.arange(0.0, 1.05, step=0.05)
    
    avg_stdev = np.mean([data[f'{comp}_stdev'] for comp in ['Bx', 'By', 'Bz']], axis=0)
    avg_vol = np.mean([data[f'{comp}_vol'] for comp in ['Bx', 'By', 'Bz']], axis=0)

    counts, xedges, yedges, image = plt.hist2d(
        avg_stdev,
        avg_vol,
        bins=[bins, bins],
        cmap='plasma',
        edgecolor='black',
        linewidth=0.5
    )
    
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcenters, ycenters)

    plt.contour(X, Y, counts.T, levels=0, colors='white', linewidths=1)

    plt.xlabel('Average Rolling Standard Deviation of Max Bx, By, Bz')
    plt.ylabel('Average Conditional Volatility of Max Bx, By, Bz')
    plt.title(f'{event_type} Events')
    plt.axis('square')
    plt.colorbar(image, label='Counts')
    plt.tight_layout()
    plt.show()
