VIDEO_NAME="${1:-01.폭행(assult)__insidedoor_07__412-4__412-4_cam02_assault01_place09_night_spring_resized}"
GPU="${2:-0}"
echo "Running our inference server"
python inference_stream.py --video_name ${VIDEO_NAME} --gpu $GPU

cd ..
echo "--------------------"
echo "Running the default tester"
python test_net.py --config_file configs/ABB/LSTR/lstr_long_256_work_8_kinetics_1x.yaml --gpu $GPU MODEL.CHECKPOINT checkpoints/ABB/LSTR/lstr_long_256_work_8_kinetics_1x/epoch-25.pth MODEL.LSTR.INFERENCE_MODE stream DATA.TEST_SESSION_SET "['${VIDEO_NAME}']"