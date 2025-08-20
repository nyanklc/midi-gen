import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# https://github.com/Yikai-Liao/symusic
'''
@inproceedings{symusic2024,
    title={symusic: A swift and unified toolkit for symbolic music processing},
    author={Yikai Liao, Zhongqi Luo, et al.},
    booktitle={Extended Abstracts for the Late-Breaking Demo Session of the 25th International Society for Music Information Retrieval Conference},
    year={2024},
    url={https://ismir2024program.ismir.net/lbd_426.html#lbd},
}
'''
from symusic import Score

DATASET_PATH = "./data/maestro-v3.0.0/"
DATASET_PROCESSED_PATH = "./data/maestro-v3.0.0-processed/"
DUMMY = "./data/dummy/"

class MIDIDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.files = []
        self.file_names = []

        for dirpath, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.mid', '.midi')):
                    self.files.append(os.path.join(dirpath, file))
                    self.file_names.append(file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        try:
            score = Score(file_path)
            # print(f"loaded score: {score}")
            # print(f"tracks: {score.tracks}")
            # print(f"tpq: {score.tpq}")
            # print(f"clipped: {score.clip(0, 100)}")
        except Exception as e:
            raise RuntimeError(f"Could not load {file_path}: {e}")

        score = score.resample(tpq=8, min_dur=1)
        # print(f"resampled score: {score}")

        # returned shape: [modes, tracks, pitch, time]
        #
        # modes: onset: Only the beginning (attack) of each note is marked
        #        frame: The entire duration of each note is marked
        #        offset: Only the end (release) of each note is marked
        pianoroll = score.pianoroll(
            modes=["onset", "frame", "offset"],
            pitch_range=[21, 109],
            encode_velocity=False
        )
        # print(f"pianoroll: {pianoroll} shape: {pianoroll.shape}")

        # # Visualize the piano roll
        # plt.figure(figsize=(12, 6))
        # plt.imshow(pianoroll[0, 0] + pianoroll[1, 0], origin='lower', aspect='auto',
        #         extent=[0, pianoroll.shape[3], 0, 128])
        # plt.title('Piano Roll (Track 0)')
        # plt.xlabel('Time (in ticks)')
        # plt.ylabel('Pitch (MIDI note number)')
        # plt.colorbar(label='Velocity')
        # plt.show()

        # score.dump_midi(DUMMY + "out_score.midi")
        # score_resampled.dump_midi(DUMMY + "out_score_resampled.midi")

        # combine the roll modes, take only the first track
        pianoroll = pianoroll[0, 0, :, :] + pianoroll[1, 0, :, :] + pianoroll[2, 0, :, :]
        print(f"{file_path} - pianoroll: {pianoroll} shape: {pianoroll.shape}")

        return pianoroll, self.file_names[idx]

def data_loader():
    for dirpath, _, files in os.walk(DATASET_PROCESSED_PATH):
        for file in files:
            pass # TODO

def process_dataset():
    d = MIDIDataset(DATASET_PATH)
    for i in range(len(d)):
        notes, filename = d.__getitem__(i)
        filename, _ = os.path.splitext(filename)
        notes.dump(DATASET_PROCESSED_PATH + filename + ".npy")

def dummy():
    d = MIDIDataset(DUMMY)

    notes = d.__getitem__(0)
    print("----------------------------")
    notes = d.__getitem__(1)
    print("----------------------------")
    notes = d.__getitem__(2)
    print("----------------------------")
    notes = d.__getitem__(3)
    print("----------------------------")
    notes = d.__getitem__(4)
    print("----------------------------")
    notes = d.__getitem__(5)

if __name__ == "__main__":
    process_dataset()
