import torch


def compute_fischer(net, dl, loss_fn, optimizer, device, fim_list=None):
    train_loader = iter(dl)

    params_w_grad = [param for param in net.parameters() if param.requires_grad]
    if fim_list is None:
        fim_list = [[torch.zeros_like(param)] for param in params_w_grad]
    else:
        for i, param in enumerate(params_w_grad):
            fim_list[i].append(torch.zeros_like(param))

    for (inputs, label) in train_loader:  # training loop
        net.train()
        net.reset_()

        inputs = inputs.view(inputs.shape[0], -1, inputs.shape[-1]).to(device)
        label = label.to(device)

        inputs = net.init_state(inputs)
        for t in range(inputs.shape[-1]):
            spikes, readouts, voltages = net(inputs[..., t].unsqueeze(-1))
            loss = loss_fn(readouts, voltages, label)
            loss.backward()

        for i, param in enumerate(params_w_grad):
            fim_list[i][-1] += param.grad.data.pow(2)

        optimizer.zero_grad()
    return fim_list


def update_previous_params_list(net, previous_params=None):
    params_w_grad = [param for param in net.parameters() if param.requires_grad]
    if previous_params is None:
        previous_params = [[param.detach().clone()] for param in params_w_grad]
    else:
        for i, param in enumerate(params_w_grad):
            previous_params[i].append(param)

    return previous_params
    

def ewc_reg(net, previous_params, fim_list):
    params_w_grad = [param for param in net.parameters() if param.requires_grad]
    reg = 0

    if fim_list is not None:
        for i, param in enumerate(params_w_grad):
            for k in range(len(fim_list[0])):
                reg += torch.sum(fim_list[i][k] * (previous_params[i][k] - param) ** 2)

    return reg

