import numpy as np
import json
import spacy
import open_clip

nlp = spacy.load("en_core_web_sm")

def invert_dictionary(d):
    inv_map = {v: k for k, v in d.items()}
    return inv_map


def read_dict(file_path):
    with open(file_path) as fin:
        return json.load(fin)


def check_segmented_object_order(scans):
    """ check all scan objects have the three_d_objects sorted by id
    :param scans: (dict)
    """
    for scan_id, scan in scans.items():
        idx = scan.three_d_objects[0].object_id
        for o in scan.three_d_objects:
            if not (o.object_id == idx):
                print('Check failed for {}'.format(scan_id))
                return False
            idx += 1
    return True


def sample_scan_object(object, scans, n_points):
    sample = object.sample(n_samples=n_points, scans=scans)
    return np.concatenate([sample['xyz'], sample['color']], axis=1)


def pad_samples(samples, max_context_size, padding_value=1):
    n_pad = max_context_size - len(samples)

    if n_pad > 0:
        shape = (max_context_size, samples.shape[1], samples.shape[2])
        temp = np.zeros(shape, dtype=samples.dtype) * padding_value
        temp[:samples.shape[0], :samples.shape[1]] = samples
        samples = temp

    return samples


def semantic_labels_of_context(context, max_context_size, label_to_idx, ins_sem_mapping,add_padding=True):
    ori_semantic_labels = [ins_sem_mapping[i.instance_label] for i in context]
    if add_padding:
        n_pad = max_context_size - len(context)
        ori_semantic_labels.extend(['pad'] * n_pad)
    assert label_to_idx is not None
    semantic_labels = np.array(
        [label_to_idx[x] if x in label_to_idx else label_to_idx["other"] for x in ori_semantic_labels])
    return semantic_labels


def instance_labels_of_context(context, max_context_size, label_to_idx, add_padding=True):
    ori_instance_labels = [i.instance_label for i in context]

    if add_padding:
        n_pad = max_context_size - len(context)
        ori_instance_labels.extend(['pad'] * n_pad)

    assert label_to_idx is not None
    instance_labels = np.array(
        [label_to_idx[x] if x in label_to_idx else label_to_idx["others"] for x in ori_instance_labels])

    return instance_labels


def decode_stimulus_string(s):
    """
    Split into scene_id, instance_label, # objects, target object id,
    distractors object id.

    :param s: the stimulus string
    """
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = \
            s.split('-', maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = \
            s.split('-', maxsplit=4)

    instance_label = instance_label.replace('_', ' ')
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
    assert len(distractors_ids) == n_objects - 1

    return scene_id, instance_label, n_objects, target_id, distractors_ids


def spacy_isin_text(classnames, tokens):
    le_text1 = ' '+get_lemmatization(tokens)+' '
    le_text2 = ' '+' '.join(tokens)+' '
    res = [classname for classname in classnames if ' '+classname+' ' in le_text1 or ' '+classname+' ' in le_text2
           or ' '+classname.replace(' ', '')+' ' in le_text1 or ' '+classname.replace(' ', '')+' ' in le_text2]
    return res

def get_lemmatization(word_list):
    lemmatized_list = []
    for word in word_list:
        doc = nlp(word)
        lemmatized_word = doc[0].lemma_
        lemmatized_list.append(lemmatized_word)
    return ' '.join(lemmatized_list)


