import torch
import numpy as np
import torch.nn as nn

from matplotlib import pyplot as plt


from .local_complexity import (get_ortho_hull_around_samples, 
                               get_layer_intersections_batched, 
                               get_hull_around_samples, 
                               get_intersections_for_hulls,
                               flatten_model)





def plot_partitions(partition_images_set,  normalize_each=False, title='', figsize=(10,15), dpi=200):
    """_summary_

    Args:
        partition_images_set (_type_): _description_
        normalize_each (bool, optional): _description_. Defaults to False.
        title (str, optional): _description_. Defaults to ''.
        figsize (tuple, optional): _description_. Defaults to (10,15).
        dpi (int, optional): _description_. Defaults to 200.
    """
    num_layers = len(partition_images_set)
    
    def normalize(x, normalize_each):
        return ( (x - x.min())/(x.max() - x.min()) ) if normalize_each else x
    
    plt.figure(figsize=figsize, dpi=dpi)
    for idx, partition_image in enumerate(partition_images_set):
        plt.subplot(1, num_layers, idx + 1)
        plt.imshow(normalize(partition_image.T, normalize_each), cmap='Blues')
        if not normalize_each:
            plt.colorbar()
        plt.title(f'Layer {idx}')
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_partitions(partitions,  normalize_each=False, title='', figsize=(10,15), dpi=100,):
    """_summary_

    Args:
        partitions (_type_): _description_
        normalize_each (bool, optional): _description_. Defaults to False.
        title (str, optional): _description_. Defaults to ''.
        figsize (tuple, optional): _description_. Defaults to (10,15).
        dpi (int, optional): _description_. Defaults to 200.
    """
    num_layers = partitions.shape[-1]
    partition_images_set = [partitions[...,i] for i in range(num_layers)]
    plot_partitions(partition_images_set, normalize_each, title, figsize, dpi)


class partitions:
    """_summary_

    Returns:
        _type_: _description_
    """

    model_flatten_func = flatten_model # can be updated with a custom function
    r = 0.0019 # default value for the radius of the ortho hull
    n = 4 # default value for the number of samples in the ortho hull
    device = torch.device('cuda' ) if torch.cuda.is_available() else torch.device('cpu')
    show_partitions = show_partitions
    
    @staticmethod
    def get_partitions_from_inr(
        x_bounds, y_bounds,
        model,
        signal_dims,
        sampled_points,
        sampled_points_batch_sizes = 256) :
        """_summary_

        Args:
            x_bounds (_type_): _description_
            y_bounds (_type_): _description_
            model (_type_): _description_
            signal_dims (_type_): _description_
            sampled_points (_type_): _description_
            sampled_points_batch_sizes (int, optional): _description_. Defaults to 256.

        Returns:
            _type_: _description_
        """

        names, modules = partitions.model_flatten_func(model)
        target_ids = np.asarray([i for i,each in enumerate(modules) if (type(each)==torch.nn.modules.Linear)])

        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach() 
            return hook

        for each in target_ids:
            modules[each].register_forward_hook(get_activation(names[each]))

        layer_names = np.sort(np.asarray(names)[target_ids]) # TODO: replace with a non-sort based method but a better traversal of model tree
        
        x_linspace = torch.linspace(x_bounds[0], x_bounds[1], sampled_points).cpu()
        y_linspace = torch.linspace(y_bounds[0], y_bounds[1], sampled_points).cpu()


        xx, yy = torch.meshgrid(x_linspace, y_linspace)
        coords = torch.hstack([xx.reshape(-1,1), yy.reshape(-1,1)]).float()
        coords = coords.to(partitions.device)

        coords_batched = torch.split(coords, sampled_points_batch_sizes)

        hulls = []
        for idx, coord_batch in enumerate(coords_batched):
            hull = get_ortho_hull_around_samples(torch.Tensor(coord_batch[...,None,None]), r=partitions.r, n=partitions.n).squeeze()
            hulls.append(hull)

        hulls = torch.cat(hulls, dim=0)
        print(hulls.shape)

        intersections, p_intersections = get_intersections_for_hulls(
            hulls,
            model=model,
            batch_size=sampled_points_batch_sizes,
            layer_names=layer_names,
            activation_buffer=activations
        )


        intersections_reshaped = [intersections[...,i].reshape(sampled_points,sampled_points) for i in range(len(layer_names))]
        intersections = torch.dstack(intersections_reshaped).permute(1,0,2)
        print(intersections.shape)

        return intersections

