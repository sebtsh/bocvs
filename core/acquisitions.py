from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
import torch


def get_acquisition(acq_name, gp, train_y):
    if acq_name == "ei":
        return ExpectedImprovement(model=gp, best_f=torch.max(train_y))
    elif acq_name == "ucb":
        return UpperConfidenceBound(model=gp, beta=2.0)
    else:
        raise Exception("Incorrect acq_name passed to get_acquisition")
