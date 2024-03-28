urls_free_law = [
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0000.parquet",
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0001.parquet",
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0002.parquet",
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0003.parquet",
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0004.parquet",
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0005.parquet",
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0006.parquet",
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0007.parquet",
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0008.parquet",
    "https://huggingface.co/datasets/EleutherAI/pile/blob/refs%2Fconvert%2Fparquet/free_law/partial/train/0009.parquet"
]

...


    # streaming = False
    from diversity.pile_subset_urls import urls_hacker_news
    path, name, data_files = 'parquet', 'free_law/partial/train', urls_free_law
    # not changing
    batch_size = 512
    today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%Hh_%Mm_%Ss')
    run_name = f'{path} div_coeff_{num_batches=} ({today=} ({name=}) {data_mixture_name=} {probabilities=})'
    print(f'{run_name=}')

    # - Init wandb
    debug: bool = mode == 'dryrun'
    run = wandb.init(mode=mode, project="beyond-scale", name=run_name, save_code=True)
    wandb.config.update({"num_batches": num_batches, "path": path, "name": name, "today": today, 'probabilities': probabilities, 'batch_size': batch_size, 'debug': debug, 'data_mixture_name': data_mixture_name, 'streaming': streaming, 'data_files': data_files})
    # run.notify_on_failure() # https://community.wandb.ai/t/how-do-i-set-the-wandb-alert-programatically-for-my-current-run/4891
    print(f'{debug=}')
    print(f'{wandb.config=}')

    # -- Get probe network
    from datasets import load_dataset
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    probe_network = probe_network.to(device)

    # -- Get data set
    def my_load_dataset(path, name):
        print(f'{path=} {name=} {streaming=}')
        if path == 'json' or path == 'bin' or path == 'csv':
            print(f'{data_files_prefix+name=}')
            return load_dataset(path, data_files=data_files_prefix+name, streaming=streaming, split="train").with_format("torch")
        elif path == 'parquet':
            print(f'{data_files=}')
            return load_dataset(path, data_files=data_files, streaming=streaming, split="train").with_format("torch")
        else:
            return load_dataset(path, name, streaming=streaming, split="train").with_format("torch")
    # - get data set for real now
    if isinstance(path, str):
        dataset = my_load_dataset(path, name)
    else:
        print('-- interleaving datasets')
        datasets = [my_load_dataset(path, name).with_format("torch") for path, name in zip(path, name)]
        [print(f'{dataset.description=}') for dataset in datasets]
        dataset = interleave_datasets(datasets, probabilities)
    print(f'{dataset=}')
    batch = dataset.take(batch_size)
    print(f'{next(iter(batch))=}')
    column_names = next(iter(batch)).keys()
    print(f'{column_names=}')

    # - Prepare functions to tokenize batch
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    remove_columns = column_names  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader
    def map(batch):
        return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    tokenized_batch = map(batch)
    print(f'{next(iter(tokenized_batch))=}')
