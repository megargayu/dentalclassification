import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from dataloader.dataset import image_datasets

def run_phase(dataloaders, model, loss_fn, optimizer, device, phase):
    # Set the model's mode
    if phase == "train":
        model.train()
    else:
        model.eval()

    # Used to calculate loss and accuracy
    running_loss = 0.0
    running_corrects = 0

    # Run the epoch
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(torch.int64).to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(image_datasets[phase])
    epoch_acc = running_corrects.double() / len(image_datasets[phase])

    print("{} loss: {:.4f}, acc: {:.4f}".format(phase,
                                                epoch_loss,
                                                epoch_acc))

    if phase == "test":
      print("precision: {:.4f}, recall: {:.4f}, f1-score: {:.4f}".format(
          precision_score(labels.data.cpu(), preds.cpu()),
          recall_score(labels.data.cpu(), preds.cpu()),
          f1_score(labels.data.cpu(), preds.cpu())))

      print("confusion matrix:", confusion_matrix(labels.data.cpu(), preds.cpu()))
    return epoch_loss, epoch_acc

