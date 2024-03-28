import abc
import os
import json

import gdown
import lm_dataformat as lmd
from tqdm import tqdm

from .utils import *

class Dataset(abc.ABC):
    @abc.abstractmethod
    def name(self):
        """ Human-readable name of the dataset """
        pass

    @abc.abstractmethod
    def documents(self):
        """ A generator producing all documents in the dataset. """
        pass

    @abc.abstractmethod
    def clean(self):
        """ Remove any dataset files. """
        pass
    
    def size(self):
        """ Return an estimate of the dataset size. Implementations may use a faster, less accurate estimate. """

        size = sum(map(utf8len, tqdm(self.documents())))
        print('size', self.name(), size)
        return size
    
    def num_docs(self):
        """ Return an estimate of the number of documents in the dataset. Implementations may use a faster, less accurate estimate. """

        size = len(list(map(lambda x: None, tqdm(self.documents()))))
        print('docs', self.name(), size)
        return size
    
    def already_shuffled(self):
        """ Datasets where the source is already shuffled should override this to return True so that it isn't shuffled again. """
        return False


class WikipediaDataset(Dataset):
    def name(self):
        return "Wikipedia (en)"

    def _download(self):
        download('components/wikipedia_en/output/wikipedia-en.tar.gz', '87b78787f71297250bca644ab9d8e3992346eeb2e2ad91101487109e3d01e644', [
            Source('direct', 'http://eaidata.bmk.sh/data/wikipedia-en.tar.gz'),
        ], extract=True)

    def documents(self):
        self._download()

        for file in ls('components/wikipedia_en/output'):
            if not file.endswith('.json'):
                continue

            with open(file) as fh:
                ob = json.load(fh)
                yield from dummy_meta(ob)

    def clean(self):
        rm_if_exists('components/wikipedia_en')
    
    def size(self):
        return 6847462907
    
    def num_docs(self):
        return 6033151


class BookCorpusDataset(Dataset):
    def name(self):
        return "BookCorpus"

    def _download(self):
        download('components/bookcorpus/books1.tar.gz', 'e3c993cc825df2bdf0f78ef592f5c09236f0b9cd6bb1877142281acc50f446f9', [
            Source('direct', 'https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz'),
            Source('direct', 'http://battle.shawwn.com/sdb/books1/books1.tar.gz'),
        ], extract=True)

    def documents(self):
        self._download()

        return dummy_meta(map(fread, ls('components/bookcorpus/books1/epubtxt')))

    def clean(self):
        rm_if_exists('components/bookcorpus')

    def size(self):
        return 6767414779
    
    def num_docs(self):
        return 17868
    
    def already_shuffled(self):
        return True


class OpenWebTextDataset(Dataset):
    def name(self):
        return "OpenWebText"

    def _download(self):
        # todo: convert
        download_directory = "components/openwebtext"
        done_file = os.path.join(download_directory, "download.done")
        if not os.path.exists(done_file):
            os.makedirs(download_directory, exist_ok=True)
            url = "https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx"
            output_file = os.path.join(download_directory, "openwebtext.tar.xz")        
            gdown.download(url, output_file, quiet=False)
            sha256sum(output_file,'9fe39d154c5bc67da8c359415372b79510eb1e2edb0d035fe4f7fc3a732b9336')

            with open(done_file, "w") as fh:
                fh.write("done!")

    def documents(self):
        self._download()

        return dummy_meta(lmd.Reader('components/openwebtext/openwebtext').stream_data())

    def clean(self):
        rm_if_exists('components/openwebtext')

    
    def size(self):
        return 39757465434
    
    def num_docs(self):
        return 8013769


class GutenbergDataset(Dataset):
    def name(self):
        return "Gutenberg (PG-19)"

    def _download(self):
        if not os.path.exists('components/gutenberg'):
            # todo: convert after gcloud download is implemented
            sh("""
            mkdir -p components/gutenberg
            cd components/gutenberg
            virtualenv env
            . env/bin/activate
            pip install gsutil
            mkdir -p pg19_train
            gsutil -m rsync gs://deepmind-gutenberg/train ./pg19_train
            """)

    def documents(self):
        self._download()

        return dummy_meta(map(fread, ls('components/gutenberg/pg19_train')))

    def clean(self):
        rm_if_exists('components/gutenberg')
    
    def size(self):
        return 11678184672
    
    def num_docs(self):
        return 28602
    
    def already_shuffled(self):
        return True


class DMMathDataset(Dataset):
    def name(self):
        return "DM Mathematics"

    def _download(self):
        if not os.path.exists('components/dm_math'):
            # todo: convert after gcloud download is implemented
            sh("""
            mkdir -p components/dm_math
            cd components/dm_math
            virtualenv env
            . env/bin/activate
            pip install gsutil
            gsutil -m rsync gs://mathematics-dataset/ $PWD
            tar xf mathematics_dataset-v1.0.tar.gz
            """)
            sha256sum('components/dm_math/mathematics_dataset-v1.0.tar.gz', 'def638343403cb9ed60437d6b684c859dd23b72779f5cc5661b0a31e67c58576')

    def documents(self):
        self._download()

        return dummy_meta(chunk_at_even_lines(concat(
            map(
                lambda x: map(fread, ls('components/dm_math/mathematics_dataset-v1.0/train-' + x)), 
                ['easy', 'medium', 'hard'])
        ), 8192))

    def clean(self):
        rm_if_exists('components/dm_math')

    def size(self):
        return 8316165951
    
    def num_docs(self):
        return 1014997


class BibliotikDataset(Dataset):
    def name(self):
        return "Bibliotik"

    def _download(self):
        raise NotImplementedError('bibliotik temporarily unavailable')
        download('components/bibliotik/Bibliotik.jsonl.zst', '1aa43653f6de7ad074796bb6ca949beab584d91c5e188a66d994643838373b06', [
        ])

    def documents(self):
        self._download()

        yield from lmd.Reader('components/bibliotik/Bibliotik.jsonl.zst').stream_data(get_meta=True)

    def clean(self):
        rm_if_exists('components/bibliotik')

    def size(self):
        return 108404259563
    
    def num_docs(self):
        return 196640

    def already_shuffled(self):
        return True

class ArXivDataset(Dataset):
    def name(self):
        return "ArXiv"

    def _download(self):
        download('components/arxiv/arxiv.jsonl.zst', '084b894f513986076a7d97e5c323c7fa8ebef1733f151a7fbdb139c19c07b571', [
            Source('direct', 'http://eaidata.bmk.sh/data/arxiv.jsonl.zst'),
        ])

    def documents(self):
        self._download()

        return lmd.Reader('components/arxiv/arxiv.jsonl.zst').stream_data(get_meta=True)

    def clean(self):
        rm_if_exists('components/arxiv')

    def size(self):
        return 60353358395
    
    def num_docs(self):
        return 1264405
    
    def already_shuffled(self):
        return True


class PubMedDataset(Dataset):
    def name(self):
        return "PubMed Abstracts"

    def _download(self):
        download('components/pubmed/PUBMED_title_abstracts_2019_baseline.jsonl.zst', '15c26a83ac2b11378b8e6ba5a16bab92428de29bacb85709834948cfcf1f029b', [
            Source('direct', 'https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst'),
            Source('direct', 'http://eaidata.bmk.sh/data/PUBMED_title_abstracts_2019_baseline.jsonl.zst'),
        ])

    def documents(self):
        self._download()

        return lmd.Reader('components/pubmed/PUBMED_title_abstracts_2019_baseline.jsonl.zst').stream_data(get_meta=True)

    def clean(self):
        rm_if_exists('components/pubmed')

    def size(self):
        return 20684050384
    
    def num_docs(self):
        return 15518009


class HackerNewsDataset(Dataset):
    def name(self):
        return "HackerNews"

    def _download(self):
        download('components/hackernews/hn.jsonl.zst', '9fbc978c92a466b1653cd578700eeb8b417ddcf8c66c7c468d5c1d11ef82aed7', [
            Source('direct', 'http://eaidata.bmk.sh/data/hn.jsonl.zst'),
        ])

    def documents(self):
        self._download()

        return lmd.Reader('components/hackernews/hn.jsonl.zst').stream_data(get_meta=True)

    def clean(self):
        rm_if_exists('components/hackernews')
    
    def size(self):
        return 4185091916
    
    def num_docs(self):
        return 831198


class FullGithubDataset(Dataset):
    def name(self):
        return "Github"

    def _download(self):
        download('components/github/github.jsonl.zst.tar', 'f7a66e8226baf075a42628d10d8eba234460da73b0ffd300736036db9be3b3c3', [
            Source('direct', 'https://the-eye.eu/public/AI/pile_preliminary_components/github.tar'),
            Source('direct', 'http://eaidata.bmk.sh/data/github.tar'),
        ])

    def documents(self):
        self._download()

        return filter(lambda x: len(x[0]) < 100000, lmd.Reader('components/github/github.jsonl.zst.tar').stream_data(get_meta=True))

    def clean(self):
        rm_if_exists('components/github')
    
    def size(self):
        return 677143668214
    
    def num_docs(self):
        return 56626342
