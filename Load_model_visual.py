import warnings
warnings.filterwarnings("ignore")
# Look at architecture and latent spec
from utils.MvDataloaders import Get_dataloaders
from utils.MvLoad_models import load

# Visualize various aspects of the model
show = 1
datasets = ['Multi-MNIST', 'Multi-FMNIST', 'Multi-COIL-10']
settings = [[1, 2], [1, 3], [1, 3]]
d_idex = 2
show_view = 1
DATA = datasets[d_idex]
share = settings[d_idex][0]
view_num = settings[d_idex][1]
path_to_model_folder = './models/' + DATA + '.pt'
latent_spec = {"disc": [10], "cont": 10}
model = load(path=path_to_model_folder, view_num=view_num, shareAE=share, latent_spec=latent_spec)

# Print the latent distribution info
print(model.MvLatent_spec)
# Print model architecture
print(model)

# Visualize various aspects of the model
from viz.MvVisualize import Visualizer as Viz

# Create a Visualizer for the model
viz = Viz(model, view_num=view_num, show_view=show_view)
if show == 0:
    viz.save_images = True  # Return tensors instead of saving images
else:
    viz.save_images = False  # Return tensors instead of saving images

# Plot generated samples from the model
import matplotlib.pyplot as plt
import pylab

plt.figure('Plot generated samples from the model')
print("Plot generated samples from the model")
samples = viz.samples()
if show == 1:
    plt.imshow(samples.numpy()[0, :, :], cmap='gray')
    pylab.show()

# Plot traversal of single dimension
plt.figure('Plot traversal of single dimension')
print("Plot traversal of single dimension")
traversal = viz.latent_traversal_line(disc_idx=0, size=10)
if show == 1:
    plt.imshow(traversal.numpy()[0, :, :], cmap='gray')
    pylab.show()
# All latent traversals
plt.figure('All latent traversals')
print("All latent traversals")
traversals = viz.all_latent_traversals()
if show == 1:
    plt.imshow(traversals.numpy()[0, :, :], cmap='gray')
    pylab.show()

# Plot a grid of two interesting traversals
# discrete latent dimension across rows
plt.figure('Plot a grid of two interesting traversals')
print("Plot a grid of two interesting traversals")
traversals = viz.latent_traversal_grid(cont_idx=2, cont_axis=1, disc_idx=0, disc_axis=0, size=(10, 10))
if show == 1:
    plt.imshow(traversals.numpy()[0, :, :], cmap='gray')
    pylab.show()

# Reorder discrete latent to match order of digits
from viz.MvVisualize import reorder_img

if show == 1:
    ordering = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # The 9th dimension corresponds to 0, the 3rd to 1 etc...
    traversals = reorder_img(traversals, ordering, by_row=True)
    plt.figure('Reorder discrete latent to match order of digits')
    print("Reorder discrete latent to match order of digits")
    plt.imshow(traversals.numpy()[0, :, :], cmap='gray')
    pylab.show()

for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    plt.figure(i)
    print('cont_idx:' + str(i))
    traversal = viz.latent_traversal_grid(filename='cont_idx_' + str(i) + '.png', cont_idx=i, cont_axis=1, disc_idx=0, disc_axis=0, size=(10, 10))
    if show == 1:
        plt.imshow(traversal.numpy()[0, :, :], cmap='gray')
        pylab.show()

# Plot reconstructions
dataloader, _, _, _ = Get_dataloaders(batch_size=50, DATANAME=DATA + '.mat')
# Extract a batch of data
datarecon = []
if view_num == 2:
    for data1, data2, labels in dataloader:
        break
    datarecon = [data1, data2]
else:
    for data1, data2, data3, labels in dataloader:
        break
    datarecon = [data1, data2, data3]
recon = viz.reconstructions(datarecon, size=(10, 10))

plt.figure('Plot reconstructions')
print("Plot reconstructions")
if show == 1:
    plt.imshow(recon.numpy()[0, :, :], cmap='gray')
    pylab.show()

print("------END-----")
