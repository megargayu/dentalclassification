from train.runphase import run_phase

def train_model(dataloaders, model, loss_fn, optimizer, device, num_epochs=3):
    losses = {"train": [], "validation": []}
    accuracies = {"train": [], "validation": []}

    for epoch in range(num_epochs):
        try:
            print("Epoch {}/{}".format(epoch+1, num_epochs))
            print("-" * 10)

            for phase in ["train", "validation"]:
                epoch_loss, epoch_acc = run_phase(dataloaders, model, loss_fn, optimizer, device, phase)
                losses[phase].append(epoch_loss)
                accuracies[phase].append(epoch_acc)

        except KeyboardInterrupt:
            break

    return accuracies, losses
