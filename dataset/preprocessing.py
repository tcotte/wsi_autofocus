import argparse
import os
from typing import Optional, Tuple

import exif
import imutils.paths
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from natsort import index_natsorted, order_by_index
from tqdm import tqdm


def search_paired_distance(distance_af: float, delta: float, list_distances: np.ndarray, folder_xy_position: str,
                           list_images: list[str]) -> Optional[dict]:
    if distance_af + delta in list_distances:
        index_z1 = list_distances.tolist().index(distance_af)
        index_z2 = list_distances.tolist().index(distance_af + delta)

        item_data = {'xy_position': os.path.basename(folder_xy_position),
                     'z1_image': os.path.basename(list_images[index_z1]),
                     'z2_image': os.path.basename(list_images[index_z2]),
                     'z1_diff_focus': distance_af,
                     'z2_diff_focus': distance_af + delta}

        return item_data

    else:
        return None


def create_label_file(set_folder: str, set_name: str, delta: float, output_folder: str = './',
                      range: Optional[Tuple] = None) -> None:
    df = pd.DataFrame()

    for xy_position in tqdm(os.listdir(set_folder)):
        folder_xy_position = os.path.join(set_folder, xy_position)

        list_distances = []
        list_images = []
        for image in list(imutils.paths.list_images(folder_xy_position)):
            distance_af = float(exif.Image(image).make)
            list_distances.append(round(distance_af, 2))
            list_images.append(image)
            # print(f'{os.path.basename(image)} -> {distance_af:.2f}')

        indexes = index_natsorted(list_distances)
        list_distances = np.array(order_by_index(list_distances, indexes))
        list_images = order_by_index(list_images, indexes)

        # TODO verify if range works and optimize execution time
        if range is not None:
            list_distances = list_distances[np.where(list_distances > range[0])]
            list_images = [list_images[i] for i in np.where(list_distances > range[0])[0].tolist()]

            list_distances = list_distances[np.where(list_distances < range[1])]
            list_images = [list_images[i] for i in np.where(list_distances < range[1])[0].tolist()]

        pair_found = 0

        paired_results = Parallel(n_jobs=1)(
            delayed(search_paired_distance)(distance_af=distance_af, delta=delta, list_distances=list_distances,
                                            folder_xy_position=folder_xy_position, list_images=list_images) for
            distance_af in list_distances[list_distances < -delta])
        paired_results = [i for i in paired_results if i is not None]

        pair_found += len(paired_results)

        df = pd.concat([df, pd.DataFrame(data=paired_results)], ignore_index=True)

        paired_results = Parallel(n_jobs=1)(
            delayed(search_paired_distance)(distance_af=distance_af, delta=delta, list_distances=list_distances,
                                            folder_xy_position=folder_xy_position, list_images=list_images) for
            distance_af in list_distances[list_distances > 0])

        paired_results = [i for i in paired_results if i is not None]

        pair_found += len(paired_results)

        df = pd.concat([df, pd.DataFrame(data=paired_results)], ignore_index=True)

    print(f'pairs found: {pair_found} with a stack of {len(list_distances)} items')


    df.to_csv(os.path.join(output_folder, f'{set_name}.csv'))


def main(args: argparse.Namespace) -> None:
    for set_folder, set_name in zip([args.train_dataset_folder, args.test_dataset_folder], ['train', 'test']):
        create_label_file(set_folder=set_folder,
                          set_name=set_name,
                          delta=args.delta,
                          output_folder=args.output_folder,
                          range=args.limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Label files creator',
        description='The aim of this script is to compute label files suggesting pairs of images I1 and I2.')

    # Add arguments
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Chosen distance between z1 and z2."
    )
    parser.add_argument(
        "--train_dataset_folder",
        type=str,
        required=True,
        help="Train dataset folder"
    )
    parser.add_argument(
        "--test_dataset_folder",
        type=str,
        required=True,
        help="Test dataset folder"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default='../labels',
        help="Folder where label files will be created"
    )

    parser.add_argument('--limit', nargs='+', type=int,
                        help="Interval limit")

    # Parse command line arguments
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    main(args)
