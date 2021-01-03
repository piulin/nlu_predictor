import sys

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, figsize= (10,7)):
    df_cm = pd.DataFrame(cm, index = [i for i in classes],
                  columns = [i for i in classes])
    plt.figure(figsize = figsize )
    sn.heatmap(df_cm, annot=True,fmt='g')
    plt.show()

def assess(dataset, intent_true, intent_pred, slots_true, slots_pred, plot=True):



    if (len(intent_true) != len(intent_pred) or len(slots_true) != len(slots_pred)):

        print("Cannot assess. Dimension mismatch", file=sys.stderr)

    cm_intent = confusion_matrix(intent_true, intent_pred)
    labels_intent = [dataset.intent_converter.id2T(i) for i in range(dataset.intent_converter.no_entries())]
    if plot:
        plot_confusion_matrix(cm_intent, labels_intent)
    precision_intent, recall_intent, fscore_intent, _ = precision_recall_fscore_support(intent_true, intent_pred,
                                                                                        average='weighted')
    print('****----intent----*****')
    print(f"precision: {precision_intent}")
    print(f"recall: {recall_intent}")
    print(f"fscore: {fscore_intent}")

    cm_slots = confusion_matrix(slots_true, slots_pred)
    labels_slots = dataset.get_labels_slots()
    #print(labels_slots)
    if plot:
        plot_confusion_matrix(cm_slots, labels_slots, (30, 15))
    precision_slots, recall_slots, fscore_slots, _ = precision_recall_fscore_support(slots_true, slots_pred,
                                                                                     average='weighted')
    print('****---slots----****')
    print(f"precision: {precision_slots}")
    print(f"recall: {recall_slots}")
    print(f"fscore: {fscore_slots}")

    return [precision_intent, recall_intent, fscore_intent], [precision_slots, recall_slots, fscore_slots]
