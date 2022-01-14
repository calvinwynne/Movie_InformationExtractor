import re, os
import nltk
import json
import pandas as pd
nltk.download('all', quiet=True)


def ie_preprocess(document):
    '''
    This funciton returns a list of sentences tagged with part-of-speech tags for a given document. Part-of-speech (POS)
    tagging is a popular Natural Language Processing process which refers to categorizing words in a text (corpus)
    in correspondence with a particular part of speech, depending on the definition of the word and its context

    Arguments:
    ----------
    document : String  Ex: "This is a sample document"
               Input document that requires realtion extraction
    RETURN:
    -------
    out      : A list of lists that contain individual token with its respective Part-of-speech tag
    '''
    return [nltk.pos_tag(nltk.word_tokenize(each_sentence)) for each_sentence in nltk.tokenize.sent_tokenize(document)]


def find_named_entities(tagged_document, binary=False):
    '''Return a list of all the named entities in the given tagged document.'''
    named_entities, entities_tree = [], []
    for each_sentence in tagged_document:
        tree = nltk.ne_chunk(each_sentence, binary)

        for each_tree in tree.subtrees():
            if each_tree.label() == "NE":
                each_entity = ""
                for each_leaf in each_tree.leaves():
                    each_entity += each_leaf[0] + " "
                named_entities.append(each_entity.strip())
        entities_tree.append(tree)

    return named_entities, entities_tree


def relation_extractor1(document, verbose=False):

    '''
    ++++++++++++++++++++++++++++++++++++++++++ Approach 1 ++++++++++++++++++++++++++++++++++++++++++++

    Uses regex to filter out NE by matching entities from lcon, filler, untagged_filler and rcon. This
    primarily follows hearst pattern, similar to the one thought in our labs. A regex is constructed
    based on the pattern that is to be extracted, applied to a relation extracted tree generated
    from the chunked sentences, and the appropriate named_entity before and after the pattern is
    exrtacted. We are using ne_chucnk(, binary=True), as this wouldnt classify NE's into PEOPLE or
    ORGANIZATION.

    nltk.sem.relextract.semi_rel2reldict tags the sentece into lcon, subjclass, subjtext, subjsym,
    filler, untagged_filler,objclass, objtext, objsym, rcon. This conveniently allows us to look for
    patternsn in lcon, filler, rcon and extract entities from subjsym and objsym

    Arguments:
    ----------
    document : String  Ex: "This is a sample document"
               Input document that requires realtion extraction

    verbose  : Boolean Ex: True/False
               Dumps logs for debugging purposes

    RETURN:
    -------
    out      : A dictionary with the required entities as keys and its respecitve extracted NEs

    '''
    out = {"directed" : [],
           "produced" : [],
           "written " : []}

    tagged_document = ie_preprocess(document)
    _, trees = find_named_entities(tagged_document)
    print()

    for each_tree in trees:
        for cat in out:
            pattern = re.compile(r'.*'+cat+'.*', re.IGNORECASE)
            pairs = nltk.sem.relextract.tree2semi_rel(each_tree)
            reldicts = nltk.sem.relextract.semi_rel2reldict(pairs + [[[]]])
            for each_sentence in reldicts:
                if verbose:
                    for what in each_sentence:
                        print("++ ", what, " "*(17-len(what)), ":", each_sentence[what])
                    print()

            relfilter = lambda x: (pattern.match(x['filler'])
                                   or pattern.match(x['lcon'])
                                   or pattern.match(x['rcon'])
                                   or pattern.match(x['untagged_filler'])
                                  )
            rels = list(filter(relfilter, reldicts))
            for rel in rels:
                if rel['subjsym'] not in out[cat]:
                    out[cat].append(rel['subjsym'])
                if rel['objsym'] not in out[cat]:
                    out[cat].append(rel['objsym'])

        for key in out:
            if len(out[key]) == 0:
                out[key].append(None)
    return out


def relation_extractor2(doc, required_verbs, best_match=True, verbose=False):


    '''
    ++++++++++++++++++++++++++++++++++++++++++ Approach 2 ++++++++++++++++++++++++++++++++++++++++++++

    This is my take on information extraction. This funciton tries to find verbs in pos-tagged and
    chunnked setences (with binary set to False). Upon finding verbs, it runs a search in both forward
    and backward direction to find all PERSON and ORGANIZATION entities. These entitites are then
    saved along with their disctances from the said detected verb. Depending on the best_match parameter,
    the function would either return the named entities closest to the verb or will return on all NEs
    with their respective distances. By no means does the closest NE to the verb mean its the required
    entity, but its better than having to choose the first entity in the list. There can be instances
    where there are 2 entities with similar distance from the verb, one before and the other after, in
    that case, I have chosen to go with the one after, this can be changed in the algorithm. This function
    is modular and generic, purpose built to work with any kind of sentences. Required verbs is alist of
    verbs that requires extracton. This in a way allows for fuzzy search, i.e the shorter the value in
    required verbs, the more results the model would retrieve. This obviously has a precision trade-off
    as shown below;

    Direc : Directed, Director, Direct (minght not be the intended value)

    Arguments:
    ----------
    doc            : String  Ex: "This is a sample document"
                     Input document that requires realtion extraction

    required_verbs : List Ex: ["Directed", "Produced", "Written"]

    best_match     : Allows for choosing the best possible NE based on its proximity to the verb
                     or to choose all NEs in the sentence that has the required verb

    verbose        : Boolean Ex: True/False
                     Dumps logs for debugging purposes

    RETURN:
    -------
    req_dict       : A dictionary with the required entities as keys and its respecitve extracted NEs

    '''

    tagged_document = ie_preprocess(doc)
    _, trees = find_named_entities(tagged_document, binary=False)
    mapper2 = {}

    def extract_best(relation_dict):
        '''
        Allows for choosing the best entity from the given list of entities based on the proximity from the
        target verb. NEs from both directions(forward/backward) are compared. There can be instances
        where there are 2 entities with similar distance from the verb, one before and the other after, in
        that case, the model has been programmed to pick the entity after, this can be changed in the
        algorithm below.
        '''
        output = {}
        for key in relation_dict:
            pos_index, pos_value =  100, ''
            neg_index, neg_value = -100, ''
            for entity in relation_dict[key]:
                if entity[0] > 0 and entity[0] < pos_index:
                    pos_index, pos_value = entity[0], entity[1]
                if entity[0] < 0 and entity[0] > neg_index:
                    neg_index, neg_value = entity[0], entity[1]

            if pos_index < abs(neg_index) and pos_index != 100:
                output[key] = pos_value
            elif pos_index > abs(neg_index) and neg_index != -100:
                output[key] = neg_value
            else:
                # Change to neg_value to choose previous entity
                output[key] = pos_value
        return output

    for tree in trees:
        '''
        Finding all verbs in each sentences and extracting their named entities.
        I do realize it would be more efficient to check if the verb is our list of required verbs, rather
        than having to find all verbs, but since we only have 50 documents, and its interesting to see other
        verbs and their relations, I have chosen to display everything. This ofcourse is silenced by default.
        '''
        curr_verb = ''
        for index, entity in enumerate(tree):
            if isinstance(entity, tuple):
                if "V" in entity[1]:
                    mapper2[entity[0].lower().strip()] = []

                    ''' Backward Search: '''
                    for rev_inx in range(index, 0, -1):
                        rev_entity = tree[rev_inx]
                        if isinstance(rev_entity, nltk.Tree) and rev_entity.label() in ["PERSON", "ORGANIZATION"] :
                            _append = ''
                            for val in rev_entity:
                                _append += val[0] + ' '
                            mapper2[entity[0].lower().strip()].append(((index - rev_inx) * -1, _append.strip()))

                    ''' Forward Search: '''
                    for fwd_inx in range(index, len(tree)):
                        fwd_entity = tree[fwd_inx]
                        if isinstance(fwd_entity, nltk.Tree) and fwd_entity.label() in ["PERSON", "ORGANIZATION"] :
                            _append = ''
                            for val in fwd_entity:
                                _append += val[0] + ' '
                            mapper2[entity[0].lower().strip()].append((fwd_inx - index, _append.strip()))

            elif isinstance(entity, nltk.Tree) and entity.label() in ["PERSON", "ORGANIZATION"] :
                _append = ''
                for val in entity:
                    _append += val[0] + ' '
                if curr_verb != '':
                    mapper2[curr_verb].append(_append.strip())

    ''' Filtering down to a list of required verbs '''
    req_dict = {}
    for req_key in required_verbs:
        for key in mapper2:
            if req_key in key:

                if req_key not in req_dict:
                    req_dict.update({req_key : mapper2[key]})
                else:
                    req_dict[req_key] += mapper2[key]

    if verbose:
        print("Detecting all possible tags based on verbs and calculating their distance from the said verbs\n")
        for key in mapper2:
            print("==> ", key)
            for ent in mapper2[key]:
                print(ent[1], " "*(19-len(str(ent[1]))), ":",  ent[0])
            print()
        print("\n\n ++++++++++++++++ Selecting required Tags\n")

        for key in req_dict:
            print("==> ", key)
            for ent in req_dict[key]:
                print(ent[1], " "*(19-len(str(ent[1]))), ":",  ent[0])
            print()

    if best_match:
        return extract_best(req_dict)
    else:
        return req_dict


def extract_info(document, best_match=True):
    '''
    Combinding information from the chosen best entity
    Arguments:
    ----------
    document       : String  Ex: "This is a sample document"
                     Input document that requires realtion extraction

    best_match     : Allows for choosing the best possible NE based on its proximity to the verb
                     or to choose all NEs in the sentence that has the required verb
    RETURN:
    -------
    output         : A dictionary with the required entities as keys and its respecitve extracted NEs
    '''

    directed = list(relation_extractor2(document, ["directed"], best_match).values())
    produced = list(relation_extractor2(document, ["produced"], best_match).values())
    written  = list(relation_extractor2(document, ["written" ], best_match).values())

    output = {

        "Directed by": directed if len(directed) > 0 else None,
        "Produced by": produced if len(produced) > 0 else None,
        "Written by" : written  if len(written ) > 0 else None,
        "Task 4"     : list(relation_extractor2(doc, ["nominate"], best_match=False).values())[0]
    }

    return output

def calculate_metrics(matches):

    '''
    Calculates precision, reacall and f1 score;
    precision = matched entities / total number of extracted items (i.e. items extraced by model)
    recall    = matched entities / total number of gold-standard items (i.e. items in json)

    Arguments:
    ----------
    matches    : List of Tuples  Ex: [(['George Lucas'], ['George Lucas']), (['J. J. Abrams'], ['J. J. Abrams'])]
                 Each tuple represents a document. Left entity of the tuple is the predicted value and right entity
                 is the gold standard value
    RETURN:
    -------
    output     : A tuple that contains precision, recall and the f1 score

    '''
    match_count = 0
    extracted_count = len(documents)
    label_count = len(documents)
    for pred, lab in matches:
        if pred == None or len(pred) == 0:
            extracted_count -= 1
        elif lab[0] == None:
            label_count -= 1
        else:
            for label in lab:
                if pred[0].lower().strip() in label.lower().strip():
                    match_count += 1
                    break

    precision = match_count / extracted_count
    recall = match_count / label_count
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def calculate_scores(labels, predictions):
    ''' Funciton just used to organize the scores '''
    matches = {
        "Directed by" : [],
        "Produced by" : [],
        "Written by"  : []
    }
    for label, pred in zip(labels, predictions):
        matches["Directed by"].append((pred["Directed by"], label["Directed by"]))
        matches["Produced by"].append((pred["Produced by"], label["Produced by"]))
        matches["Written by"] .append((pred["Written by" ], label["Written by" ] if "Written by" in label else [None]))

    out1 = calculate_metrics(matches["Directed by"])
    out2 = calculate_metrics(matches["Produced by"])
    out3 = calculate_metrics(matches["Written by" ])

    precision = out1[0], out2[0], out3[0]
    recall    = out1[1], out2[1], out3[1]
    f1        = out1[2], out2[2], out3[2]

    return precision, recall, f1


def evaluate(labels, predictions):
    '''
    Evaluate the performance of relation extraction
    using Precision, Recall, and F1 scores.

    Args:
    labels: A list containing gold-standard labels
        predictions: A list containing information extracted from documents
    Returns:
        scores: A dictionary containing Precision, Recall and F1 scores
            for the information/relations extracted in Task 3.
    '''
    assert len(predictions) == len(labels)
    precision, recall, f1 = calculate_scores(labels, predictions)
    scores = {
        'Precision': precision, 'Recall': recall, 'F1': f1
    }
    return scores


if __name__ == "__main__":
    '''
    Storing the text documents and gold-standard labels as a list of strings
    '''
    documents, labels = [], []

    for idx in range(50):
        with open(os.path.join('movies', str(idx+1).zfill(2) + '.doc.txt')) as f:
            doc = f.read().strip()
        with open(os.path.join('movies', str(idx+1).zfill(2) + '.info.json')) as f:
            label = json.load(f)
        documents.append(doc)
        labels.append(label)

    extracted_infos = []
    for document in documents:
        extracted_infos.append(extract_info(document))
    scores = evaluate(labels, extracted_infos)
    print(pd.DataFrame(scores))
