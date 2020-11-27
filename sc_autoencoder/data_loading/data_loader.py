import tensorflow as tf
import tensorflow_datasets as tfds
import scanpy as sc


def load_train_test_data(strategy, batch_size, buffer_size, tensorflow_seed):
    # Fetch dataset with corresponding information and separate dataset
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']

    # Distribute data to devices and scale images
    BATCH_SIZE_PER_REPLICA = batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    # Ensure that seeds are set and reshuffle_each_iteration is False!
    # https://github.com/tensorflow/tensorflow/issues/38197
    train_dataset = mnist_train.map(scale).cache().shuffle(buffer_size, seed=tensorflow_seed, reshuffle_each_iteration=False).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    return train_dataset, eval_dataset


def preprocessing(adata):
    # Perform preprocessing of a anndata object
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalization and scaling: 
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Identify highly-variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)
    sc.pp.scale(adata, zero_center=True, max_value=3)
    x = adata.X
    data = tf.data.Dataset.from_tensor_slices((x, x))
    return data


def load_data(strategy, batch_size, buffer_size, tensorflow_seed):
    """
    Load a single cell dataset and for feeding to the model
    """
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()

    # preprocessing
    dataset = preprocessing(adata)

    # Distribute data to devices and scale images
    BATCH_SIZE_PER_REPLICA = batch_size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    dataset = dataset.cache().shuffle(buffer_size, seed=tensorflow_seed, reshuffle_each_iteration=False).batch(BATCH_SIZE)

    return dataset
