import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
from lib_pprpa.pprpa_util import generate_spectrum
import re, os

"""This script is useful if you have excitations from both the pp and hh
channel that contribute to the observed spectrum. For instance, an excitation
of an electron in the a1 orbital of a NV- center to the conduction band
along with the excitations of the e electrons to the same.

Run the setup and pprpa scripts in this examples directory to get the
files pp.npz and hh.npz, which contain the pp(hh)-RPA excitation energies,
respectively, and the transition dipole moments.

Combine the first (non-GS-recovering) transition from the hh channel with
the pp channel results with `python ingest.py -i 1 -d combined_results.p`
See the help message from argparse for more details `python ingest.py -h`
"""


def parse_idx(value: str):
    """
    Parse index ranges of the form:
      - "i-j" for continuous ranges
      - "p,q,r" for discontinuous values
      - Mix of both: "1-3,5,7-9"
    Returns a sorted list of unique integers.
    Raises argparse.ArgumentTypeError if malformed.
    """
    result = set()
    # Split by commas to handle multiple segments
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ArgumentTypeError(f"Empty entry in --idx: {value}")

        # Match continuous range i-j
        if re.match(r"^\d+-\d+$", part):
            start, end = map(int, part.split("-"))
            if start > end:
                raise ArgumentTypeError(
                    f"Invalid range {part}: start > end"
                )
            result.update(range(start, end + 1))

        # Match single integer
        elif re.match(r"^\d+$", part):
            result.add(int(part))

        else:
            raise ArgumentTypeError(
                f"Malformed index specifier: '{part}'"
            )

    return sorted(result)

def parse_dest(value: str):
    """
    filename.x with x={p,h}.
    """
    if value[-2:] == ".p":
        major = "pp"
    elif value[-2:] == ".h":
        major = "hh"
    else:
        raise ArgumentTypeError(f"dest filename must have suffix \".p\" or \".h\"")

    filename=value[:-2] + ".npz"
    return (filename, major)

def parse_source(value: str):
    if os.path.isfile(value):
        return value
    raise ArgumentTypeError(f"The provided/default file {value} does not exist.")

parser = ArgumentParser()
parser.add_argument('-pp', type=parse_source, help="path to npz file with pprpa spectral data (excitation energies and transition dipole moments).", default="pp.npz")
parser.add_argument('-hh', type=parse_source, help="path to npz file with hhrpa spectral data (excitation energies and transition dipole moments).", default="hh.npz")
parser.add_argument(
        "-i", "--idx",
        type=parse_idx,
        help=(
            "Index range(s) to pull from the secondary file. See the help for --dest for details on the promary vs secondary files."
            "Use i-j for continuous ranges, p,q for discontinuous values, "
            "or both (e.g. 1-3,5,7-9)."
        ),
        default=None
    )
parser.add_argument('-d', '--dest', type=parse_dest, help="path to combined file with combined data. Written as filename.x, where x={p,h}. This temporary extention specifies which data -pp or -hh is the primary data source. The data specified by the slice from the other (secondary) file will be included in the combined dataset. A `.npz` extention will be added.", default=None)
args=parser.parse_args()

if (args.idx is not None and args.dest is None) or (args.dest is not None and args.idx is None):
    parser.error("--dest and --idx are both required if either is not None")

if args.idx is not None:
    print("Parsed indices:", args.idx)

if args.dest is not None:
    filename, major = args.dest
    minor = "pp" if major == "hh" else "hh"
    print(f"Major data source: {major} (copied entirely)")
    print(f"Minor data source: {minor} (sliced data)")
    print(f"Filename: {filename}")

files = [args.pp, args.hh]
results = []
tdm_res = []
for f in files:
    # Keys: x, y, vee, tdm
    data = np.load(f)
    energies = np.abs(data["vee"])
    tdms = data["tdm"]

    # Sort arrays
    idxs = np.argsort(np.abs(energies))
    energies = energies[idxs]
    tdms = tdms[idxs]
    tdm_res.append(tdms)
    e_Hartree = energies / 27.211386
    f = 2/3 * e_Hartree * np.sum(tdms**2, axis=1)

    results.append(np.array((energies, f)))

pphh = {"pp":results[0], "hh":results[1]}
tdm_dict = {"pp":tdm_res[0], "hh":tdm_res[1]}
pp = pphh["pp"]
hh = pphh["hh"]

print("pp       hh")
print("e   f   e   f")
for i in range(pp.shape[1]):
    print(pp[:,i], hh[:,i])

if args.dest is None:
    quit()

section = pphh[minor][:,args.idx]
tdm_section = tdm_dict[minor][args.idx,:]

combined = np.hstack((section, pphh[major]))
comb_tdms = np.vstack((tdm_section, tdm_dict[major]))

ordering = np.argsort(combined, axis=1)[0] # take just the energy ordering

combined = combined[:, ordering]
comb_tdms = comb_tdms[ordering, :]
print()
print("combined result:")
print("e   f")
for i in range(combined.shape[1]):
    print(combined[:,i])

np.savez(filename, vee=combined[0], tdm=comb_tdms)
spectrum = generate_spectrum(combined[0], tdm=comb_tdms) #, save_to="combined_spectrum")

import matplotlib.pyplot as plt

plt.plot(*spectrum)
plt.show()

# Tests for the specific case described in the script's docstring
# assert np.allclose(tdm_dict[minor][1], comb_tdms[1])
# assert np.allclose(tdm_dict[major][5], comb_tdms[6])
