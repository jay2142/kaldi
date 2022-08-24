
Jacob Yatvitskiy
jay2142
08/24/2022

Speaker Recognition on VoxCeleb with Mean
Hilbert Coefficients and MFCC

Summary:

My goal is to evaluate the effectiveness of certain
modifications to the VoxCeleb V2 speaker recognition formula
presently implemented by Kaldi. Specifically, I replace the MFCC
feature presently used in the Kaldi formula with a combined
MFCC / Mean Hilbert Envelope Coefficient (MHEC) feature.
PCA is used for dimensionality reduction on this combined
feature. My implementation is in the v2_jay2142 subdirectory. 
To evaluate the results, I compare the Equal Error
Rate and Detection Cost Function Minimum metrics with those achieved by 
the implementation in the v2 directory (as well as with other approaches: see my paper)

Tools used / needed to run the code:
Python 3.7: The prep_final.sh script, which is used to compute features from the raw data,
must call my make_mhec.py script. This script requires Python 3.7 as well as the 
following libraries:
scipy=1.7.3, librosa=0.9.1, numpy=1.21.6, kaldiio=2.17.2
These can all be installed with pip.
I am also including a requirements.txt file in voxceleb/v2_jay2142 generated with pip freeze, 
in order to make the installation easier. It may contain more libraries than strictly necessary.


Executables to test the code:
All important scripts are in the voxceleb/v2_jay2142 directory.
decode_final.sh runs the final decoding stage. Specifically, it uses the PLDA model to
make predictions about the test x-vectors, and evaluates the accuracy of these predictions.
I have included a barebones data directory and a small exp directory with x-vectors and
the PLDA model so that the script can find everything it needs. 
This script can be run by calling ./decode_final.sh from within the v2_jay2142 directory.
It takes about 3 seconds to run.
The output looks like this:
EER: 4.438%
minDCF(p-target=0.01): 0.3997
minDCF(p-target=0.001): 0.6442

Main scripts that run everything:
All important scripts are in the voxceleb/v2_jay2142 directory.
prep_final.sh computes the MHEC and MFCC features. The MHEC computation takes place in the
background using nohup (kaldi's job system doesn't support Python). The consequence of this is that computation continues even after
the prep_final.sh script exits. Hence, we must wait until "ps -ef | grep make_mhec" returns nothing
before proceeding to the next script. This takes about 8 hours with 64 CPUs.
Run as follows: "./prep_final.sh"

run_final.sh combines that MHEC and MFCC features and runs the rest of the procedure. 
Specifically, it performs PCA, does more preprocessing, trains the DNN, extracts x-vectors,
learns a PLDA model, and computes results. Run as follows: "./run_final.sh"
Depending on setup, it may be necessary to run "nvidia-smi -c 3" to enter 
exclusive mode for the DNN training step.


I used the VoxCeleb dataset. Specifically, I obtained it (alongside some other code & resources)
from the Professor's folder here:
https://console.cloud.google.com/storage/browser/voxceleb_trained/voxceleb

A significant portion of my work was in implementing the MHEC algorithm. My implementation can be found in voxceleb/v2_jay2142/scripts/make_mhec.py




