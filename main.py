import csv
import os

from pose.classification.full_body_pose_embedder import FullBodyPoseEmbedder
from pose.classification.helper import BootstrapHelper
from pose.classification.pose_classifier import PoseClassifier


def dump_for_the_app():
    pose_samples_folder = 'fitness_poses_csvs_out'
    pose_samples_csv_path = 'C:/Users/nikol/git/python/PerfectRepTraining/fitness_poses_csvs_out/fitness_poses_csvs_out.csv'
    file_extension = 'csv'
    file_separator = ','

    # Each file in the folder represents one pose class.
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

    with open(pose_samples_csv_path, 'w') as csv_out:
        csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]

            # One file line: `sample_00001,x1,y1,x2,y2,....`.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_in:
                csv_in_reader = csv.reader(csv_in, delimiter=file_separator)
                for row in csv_in_reader:
                    row.insert(1, class_name)
                    csv_out_writer.writerow(row)


def generate_images_with_detection():
    bootstrap_images_in_folder = 'fitness_poses_images_in'
    bootstrap_images_out_folder = 'fitness_poses_images_out'
    bootstrap_csvs_out_folder = 'fitness_poses_csvs_out'

    bootstrap_helper = BootstrapHelper(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        csvs_out_folder=bootstrap_csvs_out_folder,
    )
    # Check how many pose classes and images for them are available.
    bootstrap_helper.print_images_in_statistics()
    # Bootstrap all images.
    # Set limit to some small number for debug.
    bootstrap_helper.bootstrap(per_pose_class_limit=None)

    # Check how many images were bootstrapped.
    bootstrap_helper.print_images_out_statistics()


def analyze_output():
    bootstrap_images_in_folder = 'fitness_poses_images_in'
    bootstrap_images_out_folder = 'fitness_poses_images_out'
    bootstrap_csvs_out_folder = 'fitness_poses_csvs_out'

    bootstrap_helper = BootstrapHelper(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        csvs_out_folder=bootstrap_csvs_out_folder,
    )

    pose_embedder = FullBodyPoseEmbedder()

    pose_classifier = PoseClassifier(
        pose_samples_folder=bootstrap_csvs_out_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    outliers = pose_classifier.find_pose_sample_outliers()
    print('Number of outliers: ', len(outliers))

    bootstrap_helper.analyze_outliers(outliers)


if __name__ == '__main__':
    # generate_images_with_detection()
    # analyze_output()
    dump_for_the_app()
