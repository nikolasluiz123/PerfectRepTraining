from pose.classification.full_body_pose_embedder import FullBodyPoseEmbedder
from pose.classification.helper import BootstrapHelper
from pose.classification.pose_classifier import PoseClassifier

if __name__ == '__main__':
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

    pose_embedder = FullBodyPoseEmbedder()

    pose_classifier = PoseClassifier(
        pose_samples_folder=bootstrap_csvs_out_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    outliers = pose_classifier.find_pose_sample_outliers()
    print('Number of outliers: ', len(outliers))
