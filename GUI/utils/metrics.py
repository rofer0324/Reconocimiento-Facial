import torch


def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append((correct_k / batch_size).item())
        return results
