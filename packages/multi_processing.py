import numpy as np
import pandas as pd
import multi_processing as mp



def linParts(numAtoms, numTreads):
    parts = np.linspace(0, numAtoms, min(numTreads, numAtoms)+1)
    parts = np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms, numThreads, upperTriang=False):
    # partition of atoms with an inner loop
    parts, numThreads_ = [0], min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part = (-1+part**.5)/2.
        parts.append(part)

    parts=np.round(parts).astype(int)
    if upperTriang:
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]), parts)
    return parts