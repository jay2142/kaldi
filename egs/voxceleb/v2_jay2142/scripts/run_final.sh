# Loosely based on the original run.sh
# Edited by Jacob Yatvitskiy
# jay2142

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


# The trials file is downloaded by local/make_voxceleb1_v2.pl.
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=export/voxceleb1
voxceleb2_root=export/voxceleb2
nnet_dir=exp/xvector_nnet_1a
musan_root=export/musan

stage=10

# Collect the data produced in prep_final.sh.
if [ $stage -le 0 ]; then
  for name in train voxceleb1_test train_aug_1m; do
    for n in {0..39}; do
      cat mhec/${name}/feats.$n.scp || exit 1
    done > mhec/${name}/feats.scp || exit 1
    cp data/${name}/utt2spk mhec/${name}
    cp data/${name}/utt2num_frames mhec/${name}
    cp data/${name}/utt2dur mhec/${name}

    fix_data_dir.sh mhec/${name}

    mkdir -p data/mhec_mfcc_${name}
    paste-feats --length-tolerance=2 scp:data/${name}/feats.scp scp:mhec/${name}/feats.scp ark,scp:data/mhec_mfcc_${name}/feats.ark,data/mhec_mfcc_${name}/feats.scp
  
    cp data/${name}/vad.scp data/mhec_mfcc_${name}/vad.scp
    cp data/${name}/spk2utt data/mhec_mfcc_${name}/spk2utt
    cp data/${name}/utt2spk data/mhec_mfcc_${name}/utt2spk
    cp data/${name}/wav.scp data/mhec_mfcc_${name}/wav.scp
    utils/validate_data_dir.sh data/mhec_mfcc_${name} --no-text
  done
fi

# Run PCA on the MFCC-MHEC data
if [ $stage -le 1 ]; then
  utils/shuffle_list.pl data/mhec_mfcc_train/feats.scp | head -n 5000 | sort | est-pca --dim=35 scp:- pca_result.mat
  for base_name in train train_aug_1m voxceleb1_test; do
    name="mhec_mfcc_"${base_name}
    mkdir -p data/${name}_pca
    transform-feats pca_result.mat scp,p:data/${name}/feats.scp ark,scp:data/${name}_pca/feats.ark,data/${name}_pca/feats.scp
    cp data/${name}/spk2utt data/${name}_pca/spk2utt
    test -f data/${name}/utt2num_frames && cp data/${name}/utt2num_frames data/${name}_pca/utt2num_frames
    cp data/${name}/utt2spk data/${name}_pca/utt2spk
    cp data/${name}/wav.scp data/${name}_pca/wav.scp
    cp data/${base_name}/vad.scp data/${name}_pca/vad.scp
    utils/fix_data_dir.sh data/${name}_pca
  done
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 2 ]; then
  utils/combine_data.sh data/train_final_combined data/mhec_mfcc_train_pca data/mhec_mfcc_train_aug_1m_pca

  # If we want to use a smaller subset of the data to train, use the following lines instead:
  # utils/combine_data.sh data/train_final_combined_orig data/mhec_mfcc_train_pca data/mhec_mfcc_train_aug_1m_pca
  # utils/subset_data_dir.sh data/train_final_combined_orig 100000 data/train_final_combined

  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/train_final_combined data/train_final_combined_no_sil exp/train_final_combined_no_sil
  utils/fix_data_dir.sh data/train_final_combined_no_sil
fi

if [ $stage -le 3 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=400
  mv data/train_final_combined_no_sil/utt2num_frames data/train_final_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_final_combined_no_sil/utt2num_frames.bak > data/train_final_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_final_combined_no_sil/utt2num_frames data/train_final_combined_no_sil/utt2spk > data/train_final_combined_no_sil/utt2spk.new
  mv data/train_final_combined_no_sil/utt2spk.new data/train_final_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_final_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_final_combined_no_sil/spk2utt > data/train_final_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_final_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_final_combined_no_sil/spk2utt > data/train_final_combined_no_sil/spk2utt.new
  mv data/train_final_combined_no_sil/spk2utt.new data/train_final_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_final_combined_no_sil/spk2utt > data/train_final_combined_no_sil/utt2spk

  utils/filter_scp.pl data/train_final_combined_no_sil/utt2spk data/train_final_combined_no_sil/utt2num_frames > data/train_final_combined_no_sil/utt2num_frames.new
  mv data/train_final_combined_no_sil/utt2num_frames.new data/train_final_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_final_combined_no_sil
fi

# Stages 6 through 8 are handled in run_xvector.sh
scripts/run_xvector_full.sh --stage $stage --train-stage -1 \
  --data data/train_final_combined_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs

if [ $stage -le 9 ]; then
  # Extract x-vectors for centering, LDA, and PLDA training.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    $nnet_dir data/mhec_mfcc_train_pca \
    $nnet_dir/xvectors_train_final

  # Extract x-vectors used in the evaluation.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
    $nnet_dir data/mhec_mfcc_voxceleb1_test_pca \
    $nnet_dir/xvectors_voxceleb1_test_final
fi

if [ $stage -le 10 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnet_dir/xvectors_train_final/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/xvectors_train_final/xvector.scp \
    $nnet_dir/xvectors_train_final/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $nnet_dir/xvectors_train_final/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_final/xvector.scp ark:- |" \
    ark:data/train/utt2spk $nnet_dir/xvectors_train_final/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/xvectors_train_final/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train_final/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_final/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnet_dir/xvectors_train_final/plda || exit 1;
fi

if [ $stage -le 11 ]; then
  $train_cmd exp/scores/log/voxceleb1_test_final_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train_final/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_final/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test_final/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_final/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train_final/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test_final/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train_final/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test_final || exit 1;
fi

if [ $stage -le 12 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test_final) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test_final $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test_final $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 3.128%
  # minDCF(p-target=0.01): 0.3258
  # minDCF(p-target=0.001): 0.5003
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.329%
  # minDCF(p-target=0.01): 0.4933
  # minDCF(p-target=0.001): 0.6168
fi
