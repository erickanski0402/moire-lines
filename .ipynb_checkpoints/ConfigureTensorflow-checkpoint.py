import tensorflow as tf

def configure():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'{len(gpus)} GPUs detected, restricting memory growth')
        # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

            
if __name__ == '__main__':
    configure()