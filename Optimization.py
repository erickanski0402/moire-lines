import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def clustering(model, model_architecture, X_train, y_train):
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
        'number_of_clusters': 16,
        'cluster_centroids_init': CentroidInitialization.LINEAR
    }

    # Cluster a whole model
    clustered_model = cluster_weights(model, **clustering_params)

    # Use smaller learning rate for fine-tuning clustered model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

    clustered_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=['accuracy'])

    clustered_model.fit(
        X_train,
        y_train,
        batch_size=500,
        epochs=1,
        validation_split=0.1)
    final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

    clustered_keras_file = f"./models/clustered_{model_architecture}_weights.h5"
    tf.keras.models.save_model(final_model, clustered_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', clustered_model)


def pruning(model, model_architecture, X_train, y_train):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 128
    epochs = 5
    validation_split = 0.1  # 10% of training set will be used for validation set.

    num_images = X_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.50,
            final_sparsity=0.80,
            begin_step=0,
            end_step=end_step
        )
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                              loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=['accuracy'])


    logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(X_train, y_train,
                          batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                          callbacks=callbacks)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    pruned_keras_file = f"./models/pruned_{model_architecture}_weights.h5"
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)
