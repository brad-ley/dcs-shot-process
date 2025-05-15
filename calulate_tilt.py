import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

class Tilt():
    def __init__(self, file, velocity):
        self.file = file
        self.data = pd.read_excel(file)
        self.time_col = next(ii for ii in self.data.columns if 'time' in ii.lower())
        self.data[self.time_col] = self.data[self.time_col] - np.min(self.data[self.time_col])
        self.data[self.time_col] *= 1e-3
        self.v = velocity
        
    def iterate_tilt_calculation(self):
        avg = 0
        outfile = Path(self.file).parent.joinpath("calculated_tilt.txt")
        outstr = ""
        for i, combo in enumerate(combinations(self.data["pin"], 3)):

            print("++++++++++++++++++++++")

            retstr, retval = self.calculate_tilt(pins=np.array(combo)) 

            if retval is not None:
                avg = retval + avg * i
                avg /= i + 1
                
            outstr += retstr + "\n"
        
        outstr += "-"*60 + "\n" + f"{'Average':<12}: {avg*1e3:>10.3f} mrad"

        outfile.write_text(outstr)
        print("-----------------------------", f"{'Average':<12}: {avg*1e3:>10.3f} mrad", sep="\n")
    
    def calculate_tilt(self, pins=(1,2,3)):
        impact = {}
        for pin in pins:
            d = self.data.loc[self.data["pin"]==pin]
            impact[pin] = np.array((d["x"], d["y"], d[self.time_col]*self.v - d["z"])).ravel()
            print(impact[pin])
            
        angles_for_plane = []
        for i in range(3):
            v1 = impact[pins[1]] - impact[pins[0]]
            v2 = impact[pins[2]] - impact[pins[0]]
            angles_for_plane.append(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            pins = np.roll(pins, 1)

        maxind = np.argmax(angles_for_plane)
        pins = np.roll(pins, maxind)
        angle_spanning_plane = angles_for_plane[maxind]

        if 30 < angle_spanning_plane*180/np.pi < 150:
            normal = np.cross(impact[pins[1]]-impact[pins[0]], impact[pins[2]]-impact[pins[0]])
            normal /= np.linalg.norm(normal)

            angle = np.arccos(normal[2])
            if np.abs(np.pi - angle) < angle:
                angle = np.abs(np.pi - angle)
            
            retstr = f"Pins {pins}: {angle*1e3:>10.3f} mrad (v1-v2 angle of {angle_spanning_plane*180/np.pi:.1f} deg)" 
            return retstr, angle

        else:
            retstr = f"Pins {pins}: {'-'*10:>15} (v1-v2 angle of {angle_spanning_plane*180/np.pi:.1f} deg)"
            angle = None

        return retstr, angle

if __name__=="__main__":
    Tilt(file=r"\\DCS100\Engineering\User_Data\2025\2025-DCS-Test-Shots\25-2-008\locations_and_times.xlsx", velocity=670).iterate_tilt_calculation()