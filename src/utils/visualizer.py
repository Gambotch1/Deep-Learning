import matplotlib.pyplot as plt

def plot_training_progress(losses, accuracies, output_path='training_plot.png'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.savefig(output_path)
    plt.show()

def plot_losses(num_epochs, train_losses, test_losses, output_path='images/losses.png'):
    plt.figure()
    plt.plot(range(num_epochs), train_losses, label='Train')
    plt.plot(range(num_epochs), test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    
def plot_accuracies(num_epochs, test_accuracies, output_path='images/accuracies.png'):
    plt.figure()
    plt.plot(range(num_epochs), test_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(output_path)
