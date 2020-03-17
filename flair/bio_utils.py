from collections import defaultdict, namedtuple
from itertools import combinations
from pathlib import Path

from lxml import etree

BioNERDataset = namedtuple('BioNERDataset',
                           ['documents', 'entities_per_document'])


def overlap(entity1, entity2):
    return range(max(entity1[0], entity2[0]), min(entity1[1], entity2[1]))


def merge_overlapping_entities(entities):
    entities = list(entities)

    entity_set_stable = False
    while not entity_set_stable:
        for e1, e2 in combinations(entities, 2):
            if overlap(e1, e2):
                merged_entity = (min(e1[0], e2[0]), max(e1[1], e2[1]))
                entities.remove(e1)
                entities.remove(e2)
                entities.append(merged_entity)
                break
        else:
            entity_set_stable = True

    return entities


def compute_token_char_offsets(tokens, sentence):
    start_idx = 0
    end_idx = 0

    indices = []
    for token in tokens:
        start_idx = end_idx
        while not sentence[start_idx].strip():
            start_idx += 1
        end_idx = start_idx + len(token)

        assert token == sentence[start_idx:end_idx]

        indices.append((start_idx, end_idx))

    return indices


class CoNLLWriter:

    def __init__(self, split_dir):
        self.split_path: Path = split_dir

    def process_dataset(self, dataset: BioNERDataset, out_dir: Path):
        train_dataset, dev_dataset, test_dataset = self.split_dataset(dataset)
        self.write_to_conll(train_dataset, out_dir/'train.conll')
        self.write_to_conll(dev_dataset, out_dir/'dev.conll')
        self.write_to_conll(test_dataset, out_dir/'test.conll')



    def write_to_conll(self, dataset: BioNERDataset, output_file: Path):
        with output_file.open('w') as f:
            for document_id in dataset.documents.keys():

                document_text = dataset.documents[document_id]
                tokens = document_text.split()
                entities = [range(*e) for e in
                            dataset.entities_per_document[document_id]]
                in_entity = False

                token_char_offsets = compute_token_char_offsets(
                    [token for token in tokens],
                    document_text)

                for (start_idx, end_idx), token in zip(token_char_offsets, tokens):

                    for entity in entities:
                        if start_idx in entity:
                            if in_entity != entity:
                                tag = 'B-Ent'
                                in_entity = entity
                            else:
                                tag = 'I-Ent'
                            break
                    else:
                        tag = 'O'
                        in_entity = False

                    f.write(' '.join([token, tag]) + '\n')
                f.write('\n')

    def split_dataset(self, dataset: BioNERDataset):
        with self.split_path.with_suffix('.train').open() as f:
            train_ids = {l.strip() for l in f if l.strip()}
        with self.split_path.with_suffix('.dev').open() as f:
            dev_ids = {l.strip() for l in f if l.strip()}
        with self.split_path.with_suffix('.test').open() as f:
            test_ids = {l.strip() for l in f if l.strip()}

        train_ids = sorted(id_ for id_ in train_ids if id_ in dataset.documents)
        dev_ids = sorted(id_ for id_ in dev_ids if id_ in dataset.documents)
        test_ids = sorted(id_ for id_ in test_ids if id_ in dataset.documents)

        train_dataset = BioNERDataset(
            documents={k: dataset.documents[k] for k in train_ids},
            entities_per_document={k: dataset.entities_per_document[k] for k in
                                   train_ids})

        dev_dataset = BioNERDataset(
            documents={k: dataset.documents[k] for k in dev_ids},
            entities_per_document={k: dataset.entities_per_document[k] for k in
                                   dev_ids})

        test_dataset = BioNERDataset(
            documents={k: dataset.documents[k] for k in test_ids},
            entities_per_document={k: dataset.entities_per_document[k] for k in
                                   test_ids})

        return train_dataset, dev_dataset, test_dataset


def bioinfer_to_internal(input):
    documents = {}
    entities_per_document = defaultdict(list)

    with open(input) as f_in:
        tree = etree.parse(f_in)
        sentence_elems = tree.xpath('//sentence')
        for sentence_id, sentence in enumerate(sentence_elems):
            sentence_id = str(sentence_id)
            token_ids = []
            token_offsets = []
            sentence_text = ""

            all_entity_token_ids = []
            entities = (sentence.xpath(".//entity[@type='Individual_protein']") +
                        sentence.xpath(".//entity[@type='Gene/protein/RNA']") +
                        sentence.xpath(".//entity[@type='Gene']") +
                        sentence.xpath(".//entity[@type='Protein_family_or_group']") +
                        sentence.xpath(".//entity[@type='Protein_complex']") +
                        sentence.xpath(".//entity[@type='DNA_family_or_group']"))
            for entity in entities:
                valid_entity = True
                entity_token_ids = set()
                for subtoken in entity.xpath('.//nestedsubtoken'):
                    token_id = '.'.join(subtoken.attrib['id'].split('.')[1:3])
                    entity_token_ids.add(token_id)

                if valid_entity:
                    all_entity_token_ids.append(entity_token_ids)

            for token in sentence.xpath('.//token'):
                token_text = ''.join(token.xpath('.//subtoken/@text'))
                token_id = '.'.join(token.attrib['id'].split('.')[1:])
                token_ids.append(token_id)
                token_offsets.append(len(sentence_text) + 1)
                sentence_text += ' ' + token_text

            documents[sentence_id] = sentence_text

            for entity_token_ids in all_entity_token_ids:
                entity_start = None
                for token_idx, (token_id, token_offset) in enumerate(
                        zip(token_ids, token_offsets)):
                    if token_id in entity_token_ids:
                        if entity_start is None:
                            entity_start = token_offset
                    else:
                        if entity_start is not None:
                            entities_per_document[sentence_id].append(
                                (entity_start, token_offset - 1))
                            entity_start = None

        return BioNERDataset(documents=documents,
                             entities_per_document=entities_per_document)
