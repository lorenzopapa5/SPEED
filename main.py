import numpy as np
from utils import get_zipped_dataset, accuracy
from dataset import NYU2_DataLoader
from augmentation import BasicPolicy
from network import create_SPEED_model
from loss import accurate_obj_boundaries_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from metrics import Metric


def process(dts_root, SIZE_TRAIN=0, SIZE_TEST=0):
    # Globals
    MAX_Dist_Meters_Clip = 10
    DATASET_TRAIN, DATASET_TEST, DATASET_VALID = [], [], []

    # DATASET SECTION
    dts_zipped_test = dts_root + 'Dataset_NYU/DenseDepth_NYU/nyu_test.zip'
    dts_zipped_train = dts_root + 'Dataset_NYU/DenseDepth_NYU/nyu_data.zip'

    # Unzip the Dataset
    zipped_train = get_zipped_dataset(dts_zipped_train, True)
    if zipped_train:
        dts_root_train_nyu = '/content/data/nyu2_train/'

    zipped_test = get_zipped_dataset(dts_zipped_test, True)
    if zipped_test:
        dts_root_test_nyu = '/content/'

    # Load Datasaet
    dataset_test_NYU = NYU2_DataLoader(dts_root_test_nyu, 'test')
    test_number_nyu = dataset_test_NYU.get_test_dataset()
    DATASET_TEST.append(dataset_test_NYU)
    SIZE_TEST += test_number_nyu

    dataset_train_NYU = NYU2_DataLoader(dts_root_train_nyu, 'train')
    train_number_nyu = dataset_train_NYU.get_dataset()
    dataset_train_NYU.shuffle_dts()
    DATASET_TRAIN.append(dataset_train_NYU)
    SIZE_TRAIN += train_number_nyu

    print('There are {} images for the Test and {} for the Train'.format(test_number_nyu, train_number_nyu))

    # Set-Up Parameters
    EPOCHS = 30
    BATCH_SIZE = 16

    img_test, img_dm_test, _ = DATASET_TEST[0].load_image(192, 48, colors_image=True, index=0)

    IMG_SHAPE = img_test.shape
    D_IMG_SHAPE = img_dm_test.shape
    COLOURS = True if img_test.shape[2] == 3 else False

    print('The image shape is {} and the depth image shape is {}\n'.format(IMG_SHAPE, D_IMG_SHAPE))
    print('Train over {} images and Test over {}\n'.format(SIZE_TRAIN, SIZE_TEST))
    print('MAX possible distance: {}m'.format(MAX_Dist_Meters_Clip))

    # Augmentatio di Defalut
    is_flip = False
    is_addnoise = False
    is_erase = False

    AUG_POLICY = BasicPolicy(color_change_ratio=0.50,
                             mirror_ratio=0.50,
                             flip_ratio=0.0 if not is_flip else 0.2,
                             add_noise_peak=0 if not is_addnoise else 20,
                             erase_ratio=-1.0 if not is_erase else 0.5)

    def get_batch(split_classes, batch_size):

        while True:
            # Initialize arrays that will contain the images
            input = np.zeros((batch_size, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
            target = np.zeros((batch_size, D_IMG_SHAPE[0], D_IMG_SHAPE[1], D_IMG_SHAPE[2]))

            dataset_Path = DATASET_TRAIN

            for i in range(batch_size):
                img, depth_img, _ = dataset_Path[0].load_image(IMG_SHAPE[0], D_IMG_SHAPE[0], COLOURS)
                img, depth_img = AUG_POLICY(img, depth_img)
                input[i, :, :, :] = np.expand_dims(img, axis=0)
                target[i, :, :, :] = np.expand_dims(depth_img, axis=0)

            yield input, target

    # Build Network
    model = create_SPEED_model(input_shape=img_test.shape)

    model.compile(loss=accurate_obj_boundaries_loss,
                  metrics=[accuracy],
                  optimizer=Adam(learning_rate=1e-4, amsgrad=True))
    print('Model Compiled, ready to Train\n\n')

    if True:
        model.summary()

    # Model Fit
    save_callback = [ModelCheckpoint('/content/best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)]

    model.fit(get_batch('train', BATCH_SIZE), shuffle=True, epochs=EPOCHS, verbose=1,
              steps_per_epoch=SIZE_TRAIN // BATCH_SIZE, callbacks=[save_callback])


def evaluate(model):
    ev = Metric(model, 8, 150, dts=DTS)  # ???
    dic_acc_results = ev.update()
    ev.display_avg()


if __name__ == '__main__':
    # Training
    root = '/content/drive/MyDrive/Tesi/'
    dts_root = root + 'Datasets/'
    process(dts_root=dts_root)

    # Load Trained model
    log_event = root + 'Models/Models_Giugno/Test_PAPER_ICRA2022/Test_PPMobileNetMSPP_PT3/'
    model = load_model(log_event + 'best_model.h5', compile=False)
    print('Model loaded\n')

    # Evaluate
    evaluate(model=model)
