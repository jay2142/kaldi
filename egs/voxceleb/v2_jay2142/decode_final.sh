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

stage=11

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

