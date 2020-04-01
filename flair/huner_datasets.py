import os
from collections import namedtuple, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Union, Callable, Optional, Dict
import shutil
from abc import ABC, abstractmethod

from lxml import etree

import flair
from flair.datasets import ColumnCorpus
from flair.file_utils import cached_path, unzip_file

InternalHUNERDataset = namedtuple('BioNERDataset',
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

    def __init__(self, split_path=None):
        self.split_path: Optional[Path] = split_path

    def process_dataset(self, datasets: Dict[str, InternalHUNERDataset], out_dir: Path):
        train_dataset, dev_dataset, test_dataset = self.split_dataset(datasets['all'])
        self.write_to_conll(train_dataset, out_dir / 'huner' / 'train.conll')
        self.write_to_conll(dev_dataset, out_dir / 'huner' / 'dev.conll')
        self.write_to_conll(test_dataset, out_dir / 'huner' / 'test.conll')

        if 'train' in datasets and 'dev' in datasets and 'test' in datasets:
            self.write_to_conll(datasets['train'], out_dir / 'default' / 'train.conll')
            self.write_to_conll(datasets['dev'], out_dir / 'default' / 'dev.conll')
            self.write_to_conll(datasets['test'], out_dir / 'default' / 'test.conll')
        else:
            self.write_to_conll(train_dataset, out_dir / 'default' / 'train.conll')
            self.write_to_conll(dev_dataset, out_dir / 'default' / 'dev.conll')
            self.write_to_conll(test_dataset, out_dir / 'default' / 'test.conll')

    def write_to_conll(self, dataset: InternalHUNERDataset, output_file: Path):
        os.makedirs(output_file.parent, exist_ok=True)
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

    def split_dataset(self, dataset: InternalHUNERDataset):
        with self.split_path.with_suffix('.train').open() as f:
            train_ids = {l.strip() for l in f if l.strip()}
        with self.split_path.with_suffix('.dev').open() as f:
            dev_ids = {l.strip() for l in f if l.strip()}
        with self.split_path.with_suffix('.test').open() as f:
            test_ids = {l.strip() for l in f if l.strip()}

        train_ids = sorted(id_ for id_ in train_ids if id_ in dataset.documents)
        dev_ids = sorted(id_ for id_ in dev_ids if id_ in dataset.documents)
        test_ids = sorted(id_ for id_ in test_ids if id_ in dataset.documents)

        train_dataset = InternalHUNERDataset(
            documents={k: dataset.documents[k] for k in train_ids},
            entities_per_document={k: dataset.entities_per_document[k] for k in
                                   train_ids})

        dev_dataset = InternalHUNERDataset(
            documents={k: dataset.documents[k] for k in dev_ids},
            entities_per_document={k: dataset.entities_per_document[k] for k in
                                   dev_ids})

        test_dataset = InternalHUNERDataset(
            documents={k: dataset.documents[k] for k in test_ids},
            entities_per_document={k: dataset.entities_per_document[k] for k in
                                   test_ids})

        return train_dataset, dev_dataset, test_dataset


class HUNERDataset(ColumnCorpus, ABC):
    """
    Base class for HUNER Datasets
    TODO Docs only for development. Have to be rewritten but I don't know the format.

    Every subclass has to implement `to_internal' that produces
    a dictionary of InternalHUNERDatasets.
    dict['all'] contains the unsplitted dataset

    If splits are available then the following datasets are present additionally:
    dict['train'] -> train split
    dict['dev'] -> development split
    dict['test'] -> test split

    If splits are not available then the HUNER splits are used as default splits.
    """

    @staticmethod
    @abstractmethod
    def to_internal(data_folder: Path) -> Dict[str, InternalHUNERDataset]:
        pass

    @property
    @abstractmethod
    def split_url(self):
        pass

    def __init__(
            self,
            use_huner_split: bool = False,
            base_path: Union[str, Path] = None,
            tag_to_bioes: str = "ner",
            in_memory: bool = True
    ):
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "ner"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        split_name = self.split_url.split('/')[-1]

        train_file_default = data_folder / 'default' / 'train.conll'
        dev_file_default = data_folder / 'default' / 'dev.conll'
        test_file_default = data_folder / 'default' / 'test.conll'

        train_file_huner = data_folder / 'huner' / 'train.conll'
        dev_file_huner = data_folder / 'huner' / 'dev.conll'
        test_file_huner = data_folder / 'huner' / 'test.conll'

        if not (train_file_default.exists() and dev_file_default.exists()
                and test_file_default.exists() and
                train_file_huner.exists() and dev_file_huner.exists() and
                test_file_huner.exists()):
            cached_path(self.split_url + '.train', data_folder / 'split')
            cached_path(self.split_url + '.dev', data_folder / 'split')
            cached_path(self.split_url + '.test', data_folder / 'split')
            writer = CoNLLWriter(data_folder / 'split' / split_name)
            internal_datasets = self.to_internal(data_folder)
            writer.process_dataset(internal_datasets, data_folder)

        if use_huner_split:
            super(HUNERDataset, self).__init__(
                data_folder / 'huner', columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
            )
        else:
            super(HUNERDataset, self).__init__(
                data_folder / 'default', columns, tag_to_bioes=tag_to_bioes, in_memory=in_memory
            )


class BioInfer(HUNERDataset):
    @property
    def split_url(self):
        return "https://raw.githubusercontent.com/hu-ner/huner/master/ner_scripts/splits/bioinfer"

    @staticmethod
    def to_internal(data_path: Path) -> Dict[str, InternalHUNERDataset]:
        documents = {}
        entities_per_document = defaultdict(list)
        data_url = "http://mars.cs.utu.fi/BioInfer/files/BioInfer_corpus_1.1.1.zip"
        data_path = cached_path(data_url, data_path)
        unzip_file(
            data_path, data_path.parent
        )

        tree = etree.parse(str(data_path.parent / "BioInfer_corpus_1.1.1.xml"))
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

        return {'all': InternalHUNERDataset(documents=documents,
                                            entities_per_document=entities_per_document)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    shutil.rmtree('/Users/leon/.flair/datasets/bioinfer', ignore_errors=True)
    BioInfer()
