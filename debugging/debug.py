import torch


def get_combinations(array_1, array_2):
    res = []
    for item_1 in array_1:
        for item_2 in array_2:
            res.append([item_1, item_2])
    return res

def kernel_sum(combinations):
    k = 0
    alphas = [0.1, 1.0, 10]
    for comb in combinations:
        for alpha in alphas:
            k += torch.exp(- alpha * (comb[0], comb[1]) ** 2).sum()
    return k


def mmd_loss(y):
    """
    y: torch.Tensor of shape (batch_size * num_domains, mmd_size)
    """
    mmd_loss = 0
    y_list = []
    for i in range(len(y_list)):
        for j in range(i+1):
            combinations = get_combinations(y_list[i], y_list[j])
            term = kernel_sum(combinations)
            if i == j:
                mmd_loss += term
            else:
                mmd_loss -= 2 * term



    n0 = y_label_0.shape[0]
    n1 = y_label_1.shape[0]
    args_00 = get_combinations(y_label_0, y_label_0)
    args_11 = get_combinations(y_label_1, y_label_1)
    args_01 = get_combinations(y_label_0, y_label_1)
    term_00 = kernel(args_00) / (n0 ** 2)
    term_1 = kernel(args_1) / (n1 ** 2)
    term_01 = kernel(args_01) / (n0 * n1)
    return term_00 + term_11 - 2 * term_01

if __name__ == "__main__":
    a = torch.arange(4 * 3 * 16 * 16).view(4, 3, 16, 16)
    b = a * 10