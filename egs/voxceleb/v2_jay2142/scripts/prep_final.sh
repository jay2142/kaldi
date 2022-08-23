# Loosely based on the original run.sh
# Edited by Jacob Yatvitskiy
# jay2142

. ./cmd.sh
. ./path.sh
set -e
set -x
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


# The trials file is downloaded by local/make_voxceleb1_v2.pl.
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=export/voxceleb1
voxceleb2_root=export/voxceleb2
nnet_dir=exp/xvector_nnet_1a
musan_root=export/musan

stage=0


# Set up the train and test data
if [ $stage -le 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
  local/make_voxceleb1_v2.pl $voxceleb1_root dev data/voxceleb1_train
  local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test
  utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
fi

# Make MFCCs and compute the energy-based VAD for each dataset
if [ $stage -le 1 ]; then

  for name in train; do

    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 64 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/train_aug_orig data/train_reverb data/train_noise data/train_music data/train_babble
fi

if [ $stage -le 3 ]; then
  utils/subset_data_dir.sh data/train_aug_orig 1000000 data/train_aug_1m
  utils/fix_data_dir.sh data/train_aug_1m
  # Make MFCCs and MHECs and compute the energy-based VAD for each dataset
  for name in train_aug_1m voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 64 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}

    mkdir -p mhec/logs
    mkdir -p mhec/${name}/wav_data

    split_scp.pl data/${name}/wav.scp mhec/${name}/wav_data/wav.0.scp mhec/${name}/wav_data/wav.1.scp mhec/${name}/wav_data/wav.2.scp mhec/${name}/wav_data/wav.3.scp mhec/${name}/wav_data/wav.4.scp mhec/${name}/wav_data/wav.5.scp mhec/${name}/wav_data/wav.6.scp mhec/${name}/wav_data/wav.7.scp mhec/${name}/wav_data/wav.8.scp mhec/${name}/wav_data/wav.9.scp mhec/${name}/wav_data/wav.10.scp mhec/${name}/wav_data/wav.11.scp mhec/${name}/wav_data/wav.12.scp mhec/${name}/wav_data/wav.13.scp mhec/${name}/wav_data/wav.14.scp mhec/${name}/wav_data/wav.15.scp mhec/${name}/wav_data/wav.16.scp mhec/${name}/wav_data/wav.17.scp mhec/${name}/wav_data/wav.18.scp mhec/${name}/wav_data/wav.19.scp mhec/${name}/wav_data/wav.20.scp mhec/${name}/wav_data/wav.21.scp mhec/${name}/wav_data/wav.22.scp mhec/${name}/wav_data/wav.23.scp mhec/${name}/wav_data/wav.24.scp mhec/${name}/wav_data/wav.25.scp mhec/${name}/wav_data/wav.26.scp mhec/${name}/wav_data/wav.27.scp mhec/${name}/wav_data/wav.28.scp mhec/${name}/wav_data/wav.29.scp mhec/${name}/wav_data/wav.30.scp mhec/${name}/wav_data/wav.31.scp mhec/${name}/wav_data/wav.32.scp mhec/${name}/wav_data/wav.33.scp mhec/${name}/wav_data/wav.34.scp mhec/${name}/wav_data/wav.35.scp mhec/${name}/wav_data/wav.36.scp mhec/${name}/wav_data/wav.37.scp mhec/${name}/wav_data/wav.38.scp mhec/${name}/wav_data/wav.39.scp 

    # Create 40 processes that compute MHEC features
    for i in {0..39}
    do
      nohup python scripts/make_mhec.py scp:mhec/${name}/wav_data/wav.${i}.scp scp,ark:mhec/${name}/feats.${i}.scp,mhec/${name}/feats.${i}.ark > mhec/logs/${name}_make_mhec.${i}.log 2>&1 &
    done

    # WAIT UNTIL THE BACKGROUND PROCESSES FINISH BEFORE RUNNING run_final.sh.
    # Check if the background processes are done with: "ps -ef | grep make_mhec"
    # If nothing returns, then the background processes are done. If you see 40 processes running, then you need to wait.
    # Usually this takes around 8 hours.

    echo "Run next part when background processes finish"
    echo "You can check when they're done with:"
    echo "ps -ef | grep make_mhec"
    
  done
  exit
fi