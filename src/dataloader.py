import tensorflow as tf
import os

DATASET_PATH = "C:/SIdd temp/Project/Mlops_proj/dataset_sample"   # dataset path

if not os.path.exists(DATASET_PATH):
    print("Using SAMPLE dataset for CI")
    DATASET_PATH = "dataset_sample"


def get_train_val():

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        f"{DATASET_PATH}/Train",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224,224),
        batch_size=32,
        label_mode="categorical"
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        f"{DATASET_PATH}/Train",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224,224),
        batch_size=32,
        label_mode="categorical"
    )

    return train_data, val_data


def get_test():

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        f"{DATASET_PATH}/Test",
        image_size=(224,224),
        batch_size=1,
        shuffle=False,
        label_mode="categorical"
    )

    return test_data

if __name__ == "__main__":
    print("dataset_path: " ,DATASET_PATH)
    train_data, val_data = get_train_val()

    print(train_data)

