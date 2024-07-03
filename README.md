# RVC v2 Dataset Creation Tool
 -Download the audio from the specified YouTube video.
 -Separate the vocals from the background noise.
 -Split the vocals into smaller segments.
 -Package the processed segments into a ZIP file named dataset_my_audio.zip.
 
## Requirements

Make sure you have the following Python packages installed:
```bash
pip install yt-dlp ffmpeg-python numpy==1.26.4 librosa soundfile demucs
```

Additionally, make sure ffmpeg is installed on your system.

## Usage

To use the script, run it from the command line with the following syntax:

python process_audio.py <YouTube URL> <audio name>

    <YouTube URL>: The URL of the YouTube video you want to process.
    <audio name>: A name for the audio processing, which will be used for naming the files and folders.

## Example
```bash
python process_audio.py https://www.youtube.com/watch?v=DEqXNfs_HhY my_audio
```

## Output
The processed audio segments are stored in a directory dataset/<audio name>.
These segments are then packaged into a ZIP file named dataset_<audio_name>.zip located in the current directory.
