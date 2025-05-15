import numpy as np
import matplotlib.pyplot as plt
import isfreader as isf
import re
from pathlib import Path

pattern = re.compile(r"__S(\d+)C\d+\.isf")


class Import():
    def __init__(self, shot="", scopes=(), folder="//DCS100/Engineering/User_Data/2025/2025-DCS-Test-Shots"):
        if len(scopes) == 0:
            raise AttributeError("Need scopes! Input set, tuple, or list of scope numbers to read from.")
        if len(shot) == 0:
            raise AttributeError("Need shot number! Input shot number as string (e.g. '25-2-008').")

        self.shot = self.shot_parse(shot)
        self.scopes = set(scopes)
        
        self.experiment_folder = next(ii for ii in Path(rf"{folder}").iterdir() if self.shot_parse(ii.stem) == self.shot)
        
    def shot_parse(self, shot):
        return [int(ii) for ii in shot.split("-")]
        
    def imp(self):
        files_to_search = []
        for file in self.experiment_folder.iterdir():
            match = pattern.search(str(file))
            if match:
                scope = int(match.group(1))
                if scope in self.scopes:
                    files_to_search.append(file)
                    
        files_to_search.sort()
        
        collected_data = []
        for idx, file in enumerate(files_to_search):
            data = isf.read_file(file)
            match = [idx for idx, dd in enumerate(collected_data) if data.shape[0] == dd.shape[0]]
            if len(match):
                collected_data[match[0]] = np.hstack((collected_data[match[0]], data[:, [1]]))
            else:
                collected_data.append(data)
                
        self.exp_files = files_to_search
        self.all_data = collected_data

        return self
    
    def shorten_data(self, save=True, save_arrival_times=True):
        self.shortened_data = []
        self.arrival_times = {}

        c = 0
        for idx, dat in enumerate(self.all_data):
            trig_highs = []
            trig_lows = []
            for idx in np.arange(1, dat.shape[1]):
                w = np.where(dat[:, idx] > np.max(dat[:, idx]) * 0.5)[0]
                trig_highs.append(w[0])
                trig_lows.append(w[-1])
                self.arrival_times[self.exp_files[c]] = dat[w[0], 0]
                c += 1
                
            width = 2 * (np.mean(trig_lows) - np.mean(trig_highs))
            idxs = (int(np.mean(trig_highs) - width), int(np.mean(trig_lows) + width))
            
            self.shortened_data.append(dat[idxs[0]:idxs[1], :])
        
        c = 0
        if save:
            for scope in self.shortened_data:
                savename = self.experiment_folder.joinpath(self.exp_files[c].stem + "_combined.txt")
                np.savetxt(savename, scope, delimiter=",")
                c += scope.shape[1]
            
        if save_arrival_times:
            savename = self.experiment_folder.joinpath("arrival_times.txt")
            outstr = f"{'Scope/Channel':<15} {'Time(s)':>10}\n"
            for (key, item) in self.arrival_times.items():
                outstr += f"{key.stem.split("__")[-1]:<15} {item:>10.4e}\n"
            savename.write_text(outstr)
        
        return self
            
    
    def plot_all(self):
        f, a = plt.subplots(nrows=len(self.shortened_data), sharex=True)
        c = 0
        for idx, dat in enumerate(self.shortened_data):
            a[idx].plot(dat[:, 0], dat[:, 1:])
            labels = [ii.stem.split("__")[-1] for ii in self.exp_files[c:c + dat.shape[1]]]
            c += dat.shape[1] - 1
            a[idx].legend(labels)
            
        
        f.supxlabel("Time (s)")
        f.supylabel("Amplitude (V)")
        plt.show()


if __name__ == "__main__":
    Import(shot="25-2-008", scopes=(29, 106)).imp().shorten_data(save=True, save_arrival_times=True).plot_all()
    