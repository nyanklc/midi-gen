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

def visualize_mat(mat):
    plt.figure(figsize=(12, 6))
    plt.imshow(mat, origin='lower', aspect='auto')
    plt.title('Roll')
    plt.xlabel('Time (in ticks)')
    plt.ylabel('Pitch (MIDI note number)')
    plt.colorbar(label='Velocity')
    plt.show()

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
        # visualize_mat(pianoroll)

        # score.dump_midi(DUMMY + "out_score.midi")
        # score_resampled.dump_midi(DUMMY + "out_score_resampled.midi")

        # combine the roll modes, take only the first track
        pianoroll = pianoroll[0, 0, :, :] + pianoroll[1, 0, :, :] + pianoroll[2, 0, :, :]
        print(f"{file_path} - pianoroll: {pianoroll} shape: {pianoroll.shape}")

        return pianoroll, self.file_names[idx]

class MatDataset(Dataset):
    # loading all the data into memory for now, shouldn't be a problem
    def __init__(self, root_dir, time_len):
        super().__init__()
        self.time_len = time_len
        self.mats = []


        # we'll use fixed time length matrices
        def chop_mat(x):
            mats = []

            if x.shape[1] <= self.time_len:
                zer = np.zeros((x.shape[0], self.time_len), dtype=np.float32)
                zer[:, :x.shape[1]] = x
                mats.append(zer)
                return mats

            i = 0
            while i < x.shape[1]:
                if i+self.time_len <= x.shape[1]:
                    mats.append(x[:, i:i + self.time_len])
                    i = i + self.time_len
                else:
                    rest = x[:, i:]
                    zer = np.zeros((rest.shape[0], self.time_len), dtype=np.float32)
                    zer[:, :rest.shape[1]] = rest
                    mats.append(zer)
                    break

            return mats


        for dirpath, _, files in os.walk(DATASET_PROCESSED_PATH):
            for file in files:
                file_full_path = dirpath + file
                x = np.load(file_full_path, allow_pickle=True)
                self.mats = self.mats + chop_mat(x)

    def __len__(self):
        return len(self.mats)

    def __getitem__(self, idx):
        return torch.tensor(self.mats[idx], dtype=torch.float32)


def process_dataset():
    d = MIDIDataset(DATASET_PATH)
    for i in range(len(d)):
        notes, filename = d.__getitem__(i)
        filename, _ = os.path.splitext(filename)
        notes.dump(DATASET_PROCESSED_PATH + filename + ".npy")

def get_processed_dataset(time_len):
    return MatDataset(DATASET_PROCESSED_PATH, time_len)

def split_dataset(dataset, ratios):
    from torch.utils.data import random_split
    total = len(dataset)
    lengths = [int(r * total) for r in ratios]
    lengths[-1] = total - sum(lengths[:-1])  # fix rounding
    return random_split(dataset, lengths)

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
    output_dir = "./out"

    d = get_processed_dataset(2000)

    os.mkdir(output_dir)
    for i in range(10):
        print(f"roll {i+1}")
        visualize_mat(d.__getitem__(i))
