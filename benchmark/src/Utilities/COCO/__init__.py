from ppq.executor.base import BaseGraphExecutor
from ppq.executor.torch import TorchExecutor
import torch
import torch.utils.data

from ppq.IR import BaseGraph

from Utilities.COCO.engine import _evaluate_any_with_coco
from Utilities.COCO.coco_utils import load_coco_from_directory


def evaluate_torch_module_with_coco(
    model: torch.nn.Module, 
    coco_validation_file_dir: str=None,
    coco_valid_annotation_file_dir: str=None,
    coco_dataloader: torch.utils.data.DataLoader=None,
    batchsize: int = 32, device: str = 'cuda',
    verbose: bool = True
):
    if not coco_dataloader:
        coco_dataloader = load_coco_from_directory(
            img_folder=coco_validation_file_dir, 
            annotation_file_path=coco_valid_annotation_file_dir,
            batchsize=batchsize
        )

    model.eval()
    return _evaluate_any_with_coco(
        model_forward_function=model,
        data_loader=coco_dataloader,
        device=device, verbose=verbose
    )
    

def evaluate_ppq_module_with_coco(
    model: torch.nn.Module,
    backbone: BaseGraph,
    coco_validation_file_dir: str,
    coco_valid_annotation_file_dir: str,
    batchsize: int = 32, device: str = 'cuda',
    verbose: bool = True
):
    class ExecutorWrapper(torch.nn.Module):
        def __init__(self, executor: BaseGraphExecutor):
            super().__init__()
            self._executor = executor
        def forward(self, x):
            return {str(idx): value for idx, value in enumerate(self._executor(x))}
    executor = TorchExecutor(graph=backbone, device=device)
    model.backbone = ExecutorWrapper(executor)
    coco_dataloader = load_coco_from_directory(
        img_folder=coco_validation_file_dir, 
        annotation_file_path=coco_valid_annotation_file_dir,
        batchsize=batchsize
    )

    return _evaluate_any_with_coco(
        model_forward_function=model, 
        data_loader=coco_dataloader, 
        device=device, verbose=verbose
    )