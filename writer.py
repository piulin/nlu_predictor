import json

from utils import detokenize


def predictions2json(dataset, intents_pred, slots_pred, outfile):
    with open(outfile, 'w') as f:
        root = {}
        idx = 0
        for stc, intent, slot in zip(dataset.stcs_literals, intents_pred, slots_pred):

            entry = {}
            entry['intent'] = dataset.intent_converter.id2T(intent)
            #             entry ['text'] = stc

            previd = -1
            val = ''

            slot_entry = {}

            for stc_idx, i in enumerate(slot):

                key = dataset.slots_converter.id2T(i)
                if key != '-':
                    if previd == i:
                        slot_entry[key] += stc[stc_idx] + ' '
                    else:

                        previd = i
                        slot_entry[key] = stc[stc_idx] + ' '

            for key, val in slot_entry.items():
                stoks = slot_entry[key].rstrip()
                slot_entry[key] = detokenize(stoks.split())

            entry['slots'] = slot_entry

            root[str(idx)] = entry
            idx += 1
        json.dump(root, f, indent=3, ensure_ascii=False)