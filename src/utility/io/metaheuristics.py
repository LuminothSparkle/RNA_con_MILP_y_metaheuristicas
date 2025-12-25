from pathlib import Path
from utility.nn import torchdefault
from utility.nn.stats import compare_archs
from utility.nn.trainer import TrainerNN

def save_archs(trainer_a: TrainerNN, trainer_b: TrainerNN, dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    dataframes = compare_archs(trainer_a.model, trainer_b.model, trainer_a.dataset)
    dataframes['comparative']['regression'].to_csv(
        path_or_buf=dir_path / 'comparative_regression.csv',
        index=True, header=True
    )
    dataframes['comparative']['class'].to_csv(
        path_or_buf=dir_path / 'comparative_class.csv',
        index=True, header=True
    )
    dataframes['first']['regression'].to_csv(
        path_or_buf=dir_path / 'first_regression.csv',
        index=True, header=True
    )
    dataframes['first']['class'].to_csv(
        path_or_buf=dir_path / 'first_class.csv',
        index=True, header=True
    )
    dataframes['second']['regression'].to_csv(
        path_or_buf=dir_path / 'second_regression.csv',
        index=True, header=True
    )
    dataframes['second']['class'].to_csv(
        path_or_buf=dir_path / 'second_class.csv',
        index=True, header=True
    )
    torchdefault.save(
        state_dict=trainer_b.state_dict(),
        f=dir_path / 'trainer.pt'
    )
