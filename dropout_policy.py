import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


class DropoutPolicy(nn.Module):
    def __init__(self, model_class, n_classes, M) -> None:
        """
        model_class: Pass a pytorch model to be used in dropout
            The model should have a fully connected final layer set with dropout layers called classifier
        """
        super().__init__()
        self.n_classes = n_classes
        self.M = M

        self.model = model_class(self.n_classes)

    def get_ensemble_outputs(self, x, use_dropout):
        return [self.model.forward(x, use_dropout=use_dropout) for i in range(self.M)]

    def forward(self, x):
        return self.model.forward(x, use_dropout=False)

    def predict(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits_list = self.get_ensemble_outputs(state, use_dropout=True)
        logits_output = self.forward(state, use_dropout=False)
        steering_classes = []
        for logits in logits_list:
            y_pred = logits.view(-1, self.n_classes)
            y_probs_pred = F.softmax(y_pred, 1)

            _, steering_class = torch.max(y_probs_pred, dim=1)
            steering_classes.append(steering_class)

        y_pred = logits_output.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0)) * 2.0 - 1.0

        return steering_cmd, torch.std(torch.FloatTensor(steering_classes))

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict({k: v for k, v in weights.items()}, strict=True)
