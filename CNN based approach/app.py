import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from skimage.transform import resize
from tensorflow import keras


# Class to average lanes with
class Lanes:
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """
    Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    small_img = resize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = resize(lane_drawn, (720, 1280, 3))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 1, dtype=cv2.CV_64F)

    return result.astype(np.uint8)


def export_video(input_location: str, output_location: str) -> None:
    """
    Exports lane annotatd video to the output location
    """

    # Load video
    clip1 = VideoFileClip(input_location)
    # Annotate the clip with lines
    vid_clip = clip1.fl_image(road_lines)
    # Save the annotated clip to the output location
    vid_clip.write_videofile(output_location, audio=False)


def preview_video(input_location: str) -> None:
    """
    Previews the video with lane annotations
    """

    cap = cv2.VideoCapture(input_location)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        annotated_frame = road_lines(frame)
        cv2.imshow("Result", annotated_frame)

    cap.release()


###################
#       DRIVER    #
###################
if __name__ == "__main__":
    MODEL_LOCATION = "full_CNN_model.h5"
    INPUT_VIDEO_LOCATION = "project_video.mp4"
    OUTPUT_VIDEO_LOCATION = "proj_reg_video.mp4"

    # Load Keras model
    model = keras.models.load_model(MODEL_LOCATION)
    # Create lanes object
    lanes = Lanes()

    # preview annotated video
    preview_video(INPUT_VIDEO_LOCATION)

    # export annotated video
    # export_video(INPUT_VIDEO_LOCATION, OUTPUT_VIDEO_LOCATION)
