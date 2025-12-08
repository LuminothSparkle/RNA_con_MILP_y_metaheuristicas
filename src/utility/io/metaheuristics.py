from pathlib import Path
from utility.nn import torchdefault
from utility.nn.stats import compare_archs
from utility.nn.trainer import TrainerNN

def save_archs(trainer_a: TrainerNN, trainer_b: TrainerNN, dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    dataframes = compare_archs(trainer_a.model, trainer_b.model)
    dataframes['regression'].to_csv(
        path_or_buf=dir_path / 'regression.csv',
        index=True, header=True
    )
    dataframes['class'].to_csv(
        path_or_buf=dir_path / 'class.csv',
        index=True, header=True
    )
    torchdefault.save(
        state_dict=trainer_b.state_dict(),
        f=dir_path / 'trainer.pt'
    )