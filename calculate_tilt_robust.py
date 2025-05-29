import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations


class Tilt:
    def __init__(self, velocity, pins=None):
        self.v = velocity * 1e3
        self.pins = pins
        self.magnification = 1

    def import_file(self, file):
        self.file = file
        self.data = pd.read_excel(file)
        self.time_col = next(ii for ii in self.data.columns if "time" in ii.lower())
        self.data[self.time_col] = self.data[self.time_col] - np.min(
            self.data[self.time_col]
        )
        self.data[self.time_col] *= 1e-3
        self.pins = self.data[
            "pin"
        ]  # this will have an error, need to convert to numpy array with rows for each pin

        return self

    def iterate_tilt_calculation(self, save_data=True):
        avg = 0
        angles = []
        normal_vecs = []
        outstr = ""
        for i, combo in enumerate(combinations(list(range(1, len(self.pins) + 1)), 3)):
            if __name__=="__main__":
                print("++++++++++++++++++++++")

            retstr, retval, normal = self.calculate_tilt(pins=np.array(combo))

            if retval is not None:
                normal_vecs.append(normal)
                angles.append(retval)
                avg = retval + avg * i
                avg /= i + 1
                
            outstr += retstr + "\n"
            
        normal_vecs = np.array(normal_vecs)
        for idx, vec in enumerate(normal_vecs):
            if vec[0] < 0:
                normal_vecs[idx, :] = -1 * vec
                # pass

        if len(angles) > 0:
            std = np.std(angles)
            avg = np.mean(angles)
            o = (
                "-" * 60
                + "\n"
                + f"{'Average':<12}: {avg * 1e3:>10.3f}+-{std * 1e3:.3f} mrad"
            )
            outstr += o

            if __name__=="__main__":
                print(o, sep="\n")
            if save_data:
                # outfile = Path(self.file).parent.joinpath("calculated_tilt.txt")
                outfile = Path("/Users/Brad/Desktop").joinpath("calculated_tilt.txt")
                outfile.write_text(outstr)

            return avg, std, normal_vecs
        
        return None, None, None
    
    def magnify_impact_axis(self, magnification=1e3):
        self.magnification = magnification
        return self

    def calculate_tilt(self, pins=(1, 2, 3)):
        pins = tuple(int(ii) for ii in pins)
        impact_coords = {}

        for pin in pins:
            # d = self.data.loc[self.data["pin"]==pin]
            # impact[pin] = np.array((d["x"], d["y"], d[self.time_col]*self.v - d["z"])).ravel()
            p = pin - 1
            impact_coords[pin] = np.array(
                (
                    self.pins[p, 0],
                    self.pins[p, 1],
                    (self.pins[p, 3] * (-self.v) - self.pins[p, 2]) * self.magnification,
                )
            )  # impact z position is the time delay*velocity minus the expected distance (measured with drop gauge)
            if __name__=="__main__":
                print(impact_coords[pin])

        angles_for_plane = []
        for i in range(3):
            v1 = impact_coords[pins[1]] - impact_coords[pins[0]]
            v2 = impact_coords[pins[2]] - impact_coords[pins[0]]
            angles_for_plane.append(
                np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            )
            pins = np.roll(pins, 1)

        maxind = np.argmax(angles_for_plane)
        pins = np.roll(pins, maxind)
        angle_spanning_plane = angles_for_plane[maxind]

        if 30 < angle_spanning_plane * 180 / np.pi < 150:
            normal = np.cross(
                impact_coords[pins[1]] - impact_coords[pins[0]],
                impact_coords[pins[2]] - impact_coords[pins[0]],
            )
            normal /= np.linalg.norm(normal)

            angle = np.arccos(normal[2])
            if np.abs(np.pi - angle) < angle:
                angle = np.abs(np.pi - angle)

            retstr = f"Pins {pins}: {angle * 1e3:>10.3f} mrad (v1-v2 angle of {angle_spanning_plane * 180 / np.pi:.1f} deg)"

        else:
            retstr = f"Pins {pins}: {'-' * 10:>15} (v1-v2 angle of {angle_spanning_plane * 180 / np.pi:.1f} deg)"
            angle = None
            normal = None

        return retstr, angle, normal


if __name__ == "__main__":
    Tilt(velocity=0.67).import_file(
        file=r"\\DCS100\Engineering\User_Data\2025\2025-DCS-Test-Shots\25-2-008\locations_and_times.xlsx"
    ).iterate_tilt_calculation(save_data=True)
