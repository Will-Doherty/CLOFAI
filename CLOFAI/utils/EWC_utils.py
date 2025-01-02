import torch
from torch.utils.data import DataLoader

def load_optimal_weights(path_to_optimal_weights):
    return torch.load(path_to_optimal_weights)

def fisher_matrix(model_f, dataset, samples, cfg):
    batch_size = samples
    data_loader_f = DataLoader(dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: torch.manual_seed(cfg.SEED))
    weights = [p for p in model_f.parameters() if p.requires_grad]
    variance = [torch.zeros_like(weight, device=cfg.device) for weight in weights]

    original_grads = [p.grad.clone().detach() if p.grad is not None else None for p in weights]
    for p in weights:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    num_batches = samples // batch_size

    for _ in range(num_batches):
        for inputs in data_loader_f:
            inputs = inputs.to(cfg.device)
            model_f.zero_grad()

            output = model_f(inputs).clamp(min=1e-9)
            log_likelihood = torch.log(output).sum(dim=1).mean()
            log_likelihood.backward()

            for weight, var in zip(weights, variance):
                if weight.grad is not None:
                    var += weight.grad.clone() ** 2

    for weight, original_grad in zip(weights, original_grads):
        weight.grad = original_grad

    fisher_diagonal = [var / (num_batches * batch_size) for var in variance]
    return fisher_diagonal

def ewc_loss(lam, model_e, dataset, samples, optimal_weights, cfg):
    lam = cfg.lambda_param
    samples = cfg.samples_replayed
    fisher_diagonal = fisher_matrix(model_e, dataset, samples, cfg)

    def loss_fn(new_model):
        loss = 0
        current_weights = [p for p in new_model.parameters() if p.requires_grad]
        for f, current, optimal in zip(fisher_diagonal, current_weights, optimal_weights):
            loss += torch.sum(f * ((current - optimal) ** 2))

        return loss * (lam / 2)

    return loss_fn