import io
import pickle

from utils import *
from batches import *
from dataset import NYU2_DataLoader, DIML_DataLoader
from augmentation import BasicPolicy
from network import *
from loss import accurate_obj_boundaries_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from metrics import Metric
from psutil import virtual_memory
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def process(dts_root, model_root, dts_type, ep, bs, lr, do_train, pretrained_path):
    # Globals
    do_print = True
    DATASET_TRAIN, DATASET_TEST = [], []
    SIZE_TRAIN, SIZE_TEST = 0, 0

    # DATASET SECTION
    if dts_type == 'nyu':
        dts_root_train_nyu = dts_root + 'NYU Dataset/nyu_train/'
        dts_root_test_nyu = dts_root + 'NYU Dataset/nyu_test/'

        # Load Datasaet
        dataset_train_NYU = NYU2_DataLoader(dts_root_train_nyu, 'train')
        train_number_nyu = dataset_train_NYU.get_dataset()
        dataset_train_NYU.shuffle_dts()
        DATASET_TRAIN.append(dataset_train_NYU)
        SIZE_TRAIN += train_number_nyu

        dataset_test_NYU = NYU2_DataLoader(dts_root_test_nyu, 'test')
        test_number_nyu = dataset_test_NYU.get_test_dataset()
        DATASET_TEST.append(dataset_test_NYU)
        SIZE_TEST += test_number_nyu

        print('STAT: There are {} Train images'.format(train_number_nyu))
        print('STAT: There are {} Test images'.format(test_number_nyu))

    elif dts_type == 'diml':
        dts_root_train_diml = dts_root + 'DIML dataset/train/'
        dts_root_test_diml = dts_root + 'DIML dataset/test/'

        dataset_train_DIML = DIML_DataLoader(dts_root_train_diml, type_dts='train')
        train_diml_number, train_lost_images = dataset_train_DIML.get_train_dataset()
        dataset_train_DIML.shuffle_dts()
        DATASET_TRAIN.append(dataset_train_DIML)
        SIZE_TRAIN += train_diml_number

        dataset_test_DIML = DIML_DataLoader(dts_root_test_diml, type_dts='test')
        test_diml_number, test_lost_images = dataset_test_DIML.get_test_dataset()
        DATASET_TEST.append(dataset_test_DIML)
        SIZE_TEST += test_diml_number

        print('STAT: There are {} Train images and {} are lost'.format(train_diml_number, train_lost_images))
        print('STAT: There are {} Test images and {} are lost\n'.format(test_diml_number, test_lost_images))

    else:
        raise ValueError('Incorrect dataset chosen.')

    # Set-Up Parameters
    img_test, img_dm_test, _ = DATASET_TEST[0].load_image(192, 48, colors_image=True, index=0)

    IMG_SHAPE = img_test.shape
    D_IMG_SHAPE = img_dm_test.shape
    COLOURS = True if img_test.shape[2] == 3 else False

    print('The image shape is {} and the depth image shape is {}'.format(IMG_SHAPE, D_IMG_SHAPE))
    print('Train over {} images and Test over {}\n'.format(SIZE_TRAIN, SIZE_TEST))

    if do_print:
        print_img(DATASET_TEST[0], img_h=IMG_SHAPE[0], d_h=D_IMG_SHAPE[0])

    exit(0)

    if do_train:
        AUG_POLICY = BasicPolicy(color_change_ratio=0.50, mirror_ratio=0.50)

        if do_print:
            print_img(DATASET_TEST[0], img_h=IMG_SHAPE[0], d_h=D_IMG_SHAPE[0], augment=True)  # TO DO

        # Build Network
        model = create_SPEED_model(input_shape=img_test.shape, existing=pretrained_path)

        model.compile(loss=accurate_obj_boundaries_loss,
                      metrics=[accuracy],
                      optimizer=Adam(learning_rate=lr, amsgrad=True))
        print('Model Compiled, ready to Train\n\n')

        if False:
            tf.keras.utils.plot_model(model, to_file=model_root + '{architecture}.png', show_shapes=True)
            print('Architecture saved in folder\n\n')

        # Callbacks
        def log_evolution_process(epoch, logs, batch_tb=4):
            x_img, gt_depth_img = get_batch_evolution(batch_tb, num_test=SIZE_TEST, dts_test=DATASET_TEST,
                                                      img_shape=IMG_SHAPE, d_shape=D_IMG_SHAPE, colour=COLOURS)
            y_pred = model.predict(x_img)

            def conversion(figure):
                """
                Converts the matplotlib plot specified by 'figure' to a PNG image and
                returns it. The supplied figure is closed and inaccessible after this call.
                """
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(figure)
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)

                return image

            # Create a figure to contain the plot.
            def plot_evolution_process(pred, gt):
                figure = plt.figure(figsize=(12, 7), dpi=100.0)
                for i in range(batch_tb):
                    _, cmap_gt, gt_min, _ = plot_depth_map(gt[i])
                    pred[i], _, _, _ = plot_depth_map(pred[i])
                    plt.subplot(2, 2, i + 1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.imshow(tf.squeeze(pred[i]), cmap=cmap_gt, vmin=gt_min)
                    plt.colorbar()
                    plt.savefig(model_root + 'evolution_images/processIMG_' + str(epoch) + '.png')

                return figure

            def plot_evolution_disparity(pred, gt):
                figure = plt.figure(figsize=(12, 7), dpi=100.0)
                for i in range(batch_tb):
                    disparity_img = tf.abs(gt[i] - pred[i])
                    plt.subplot(2, 2, i + 1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.imshow(tf.squeeze(disparity_img), cmap=plt.cm.magma)
                    plt.colorbar()
                    plt.savefig(model_root + 'evolution_images/errorIMG_' + str(epoch) + '.png')

                return figure

            fig_evolution = plot_evolution_process(y_pred, gt_depth_img)
            fig_evolution_to_store = conversion(fig_evolution)

            fig_disparity = plot_evolution_disparity(y_pred, gt_depth_img)
            fig_disparity_to_store = conversion(fig_disparity)

            with file_writer_custom.as_default():
                tf.summary.image('Evolution Image', fig_evolution_to_store, step=epoch)
                tf.summary.image('Error Image', fig_disparity_to_store, step=epoch)

        file_writer_custom = tf.summary.create_file_writer(model_root)
        save_callback = [LambdaCallback(on_epoch_end=log_evolution_process),
                         ModelCheckpoint(model_root + 'best_model.h5', monitor='val_loss', save_best_only=True,
                                         verbose=1),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.00009, min_delta=1e-2)]

        # Model Fit
        train_steps_for_epoch = SIZE_TRAIN // bs
        val_steps_for_epoch = SIZE_TEST // bs

        history = model.fit(get_batch('train', bs, dts_train=DATASET_TRAIN, dts_test=DATASET_TEST, img_shape=IMG_SHAPE,
                                      d_shape=D_IMG_SHAPE, colors=COLOURS, aug=AUG_POLICY),
                            shuffle=True, epochs=ep,
                            validation_data=get_batch('valid', bs, dts_train=DATASET_TRAIN, dts_test=DATASET_TEST,
                                                      img_shape=IMG_SHAPE, d_shape=D_IMG_SHAPE, colors=COLOURS,
                                                      aug=AUG_POLICY),
                            verbose=1,
                            steps_per_epoch=train_steps_for_epoch,
                            validation_steps=val_steps_for_epoch,
                            workers=1, use_multiprocessing=False,
                            callbacks=[save_callback])

        with open(model_root + 'trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    else:
        saved_model = 'best_model.h5'
        model = load_model(model_root + saved_model, compile=False)
        print('Model loaded\n')

        ev = Metric(model=model, bs=bs, num_test=SIZE_TEST, dts_test=DATASET_TEST, img_shape=IMG_SHAPE,
                    d_shape=D_IMG_SHAPE)
        dic_acc_results = ev.update()
        ev.display_avg()


if __name__ == '__main__':
    do_train = True
    dts_root = './PATH'
    save_model_root = './PATH'
    epochs = 30
    batch_size = 16

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(virtual_memory().total / 1e9))

    process(dts_root=dts_root, model_root=save_model_root, dts_type='nyu', ep=epochs, bs=batch_size, lr=1e-4,
            do_train=do_train, pretrained_path='')

    print('\n#-------------------#\n# Process completed #\n#-------------------#')
