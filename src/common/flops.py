from configs.globals import DEVICE
from metrics.metrics import FlopsMetrics
from utils.calc_flops.calculate_model_flops import calculate_model_flops


def calculate_flops(dataloader_category, dataloader_next, model):
    sample_category = next(iter(dataloader_category.train.dataloader))['x'].to(DEVICE)
    sample_next = next(iter(dataloader_next.train.dataloader))['x'].to(DEVICE)
    result = calculate_model_flops(model, [sample_category[1:], sample_next[1:]], print_report=True, units='K')
    return FlopsMetrics(flops=result['total_flops'], params=result['params']['total'])