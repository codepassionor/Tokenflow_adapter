import webdataset as wds
from pathlib import Path
from tqdm import tqdm
import os

video_folder_path = Path("Tokenflow_adapter/vid_edit_friendly_t2i/MSRVTT/Video/")
output_path = Path("Tokenflow_adapter/vid_edit_friendly_t2i/MSRVTT/")

max_videos_per_tar = 100

video_filenames = [f for f in os.listdir(video_folder_path) if f.endswith(".mp4")]
num_tars = len(video_filenames) // max_videos_per_tar + (1 if len(video_filenames) % max_videos_per_tar > 0 else 0)

for tar_index in tqdm(range(num_tars), desc="Creating TAR files"):
    start_index = tar_index * max_videos_per_tar
    end_index = start_index + max_videos_per_tar
    tar_video_filenames = video_filenames[start_index:end_index]
    
    tar_name = f"dataset_{tar_index:04d}.tar"
    
    with wds.TarWriter(str(output_path / tar_name)) as writer:
        for video_filename in tar_video_filenames:
            video_path = video_folder_path / video_filename
            with open(video_path, "rb") as video_file:
                sample = {
                    "__key__": video_filename,
                    "mp4": video_file.read()
                }
                writer.write(sample)
