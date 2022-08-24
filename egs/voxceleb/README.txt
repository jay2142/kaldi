
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
scipy, librosa, numpy, kaldiio
These can all be installed with pip.

Executables to test the code:
All important scripts are in the voxceleb/v2_jay2142 directory.
decode_final.sh runs the final decoding stage. Specifically, it uses the PLDA model to
make predictions about the test x-vectors, and evaluates the accuracy of these predictions.
I have included a barebones data directory and a small exp directory so that the script can find
the files it needs. 
This script can be run by calling ./decode_final.sh from within the v2_jay2142 directory.

Main scripts that run everything:
All important scripts are in the voxceleb/v2_jay2142 directory.
prep_final.sh computes the MHEC and MFCC features. The MHEC computation takes place in the
background using nohup. The consequence of this is that computation continues even after
the prep_final.sh script exits. Hence, we must wait until "ps -ef | grep make_mhec" returns nothing
before proceeding to the next script. This takes about 8 hours with 64 CPUs.

run_final.sh combines that MHEC and MFCC features and runs the rest of the procedure. 
Specifically, it performs PCA, does more preprocessing, trains the DNN, extracts x-vectors,
learns a PLDA model, and computes results. 





