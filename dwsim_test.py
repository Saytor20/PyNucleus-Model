import sys, os, clr
import pandas as pd

# 1) Point to DWSIM assemblies
mac_gac = "/Applications/DWSIM.app/Contents/MacOS/libmono/4.5"
# linux_gac = "/usr/local/lib/mono/4.5"
sys.path.append(mac_gac if os.path.isdir(mac_gac) else linux_gac)

# 2) Load DWSIM
clr.AddReference("DWSIM.Thermodynamics")
clr.AddReference("DWSIM.Interfaces")
from DWSIM.Thermodynamics.IO import DWSIMFile
from DWSIM.Thermodynamics.SimulationObjects.Streams import Stream
from DWSIM.Thermodynamics.SimulationObjects.Operations import Flash
from DWSIM.Thermodynamics.PropertyPackages import UNIFAC_LL

def run_flash():
    fs = DWSIMFile.NewEmpty()

    feed = Stream(fs, "Feed")
    feed.PropertyPackage = UNIFAC_LL()
    feed.SetTemperature(350, "K")
    feed.SetPressure(1e5, "Pa")
    feed.SetComposition({"Methanol":0.5, "Water":0.5}, "Mole")

    flash = Flash(fs, "Flash1")
    flash.PropertyPackage = UNIFAC_LL()
    fs.Connect(feed, "Outlet", flash, "Inlet")

    fs.SolveFlowsheet()

    vap = flash.OutletStreams["Vapor"]
    y_m = vap.GetProp("fraction", "Phase", "V", "", "mole")[0]

    return {"T (K)":350, "P (bar)":1, "Y_MeOH (%)": y_m*100}

if __name__ == "__main__":
    res = run_flash()
    print("► Result:", res)
    pd.DataFrame([res]).to_excel("flash_result.xlsx", index=False)
    print("► Wrote flash_result.xlsx") 