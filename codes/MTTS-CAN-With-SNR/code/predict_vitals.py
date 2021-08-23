#python code/predict_vitals.py --video_path "test-video/esra.mp4"
import matplotlib
import sklearn
import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse

import plotly.graph_objects as go

sys.path.append('../')
from model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
from scipy.signal import find_peaks, stft, lfilter, butter, welch

def predict_vitals(args):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = './mtts_can.hdf5'
    batch_size = args.batch_size
    fs = args.sampling_rate
    sample_data_path = "E:/UBFC/16.avi"

    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    #print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]
    print("video", len(dXsub))
    print("dxsublen", dXsub_len)

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    f = open("E:/UBFC/16.txt", "r")
    gtData = f.read().splitlines()
    gtData2 = np.array(gtData[1].split())
    print("GT", len(gtData2))
    gtFloat = gtData2.astype(np.float)

    gtPRvalues = []
    estPRvalues = []

    # Calculating HR With SNR formulas
    # we used the estimated values and ground truth values obtained on thirty-second video segments with the starting
    # points at one-second intervals for each video
    for i in range(30, dXsub_len - 930, 15):
        yptest = model.predict((dXsub[i:i+900, :, :, :3], dXsub[i:i+900, :, :, -3:]), batch_size=batch_size, verbose=1)
        gtPR = gtFloat[i:i+900]
        gtPRmean = np.mean(gtPR)
        gtPRvalues = np.append(gtPRvalues, gtPRmean)
        print(gtPRvalues)
        print("********************************")

        pulse_pred = yptest[0]
        pulse_pred = detrend(np.cumsum(pulse_pred), 100)
        [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

        resp_pred = yptest[1]
        resp_pred = detrend(np.cumsum(resp_pred), 100)
        [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

        pxx, frequency = matplotlib.pyplot.psd(pulse_pred, NFFT=len(pulse_pred), Fs=30, window=np.hamming(len(pulse_pred)))

        top2 = 0
        index2 = 0
        snr = []
        p = []
        fxx = []

        # Applying SNR formula
        for f in frequency:
            if f >= 1 and f <= 4:
                top2 += pxx[index2]
                p = np.append(p, pxx[index2])
                fxx = np.append(fxx, f)

            index2 = index2 + 1

        for f in frequency:
            if f >= 1 and f <= 4:
                top1 = 0
                index = 0
                for f2 in frequency:
                    if f2 >= f - 0.167/2 and f2 <= f + 0.167/2:
                        top1 += pxx[index]

                    index = index + 1

                snr_fnew = top1 / (top2 - top1)
                snr = np.append(snr, snr_fnew)

        product = np.multiply(p, snr)
        MaxInd = np.argmax(product)

        MaxValue = fxx[MaxInd]
        print("MaxValue", MaxValue)

        PR = MaxValue * 60

        estPRvalues = np.append(estPRvalues, PR)
        print(estPRvalues)

        #print("Pulse rate: ", PR)

    rmse = np.sqrt(((gtPRvalues - estPRvalues) ** 2).mean())

    print("RMSE: ", rmse)

    ########## Plot ##################
    #plt.subplot(211)
    #plt.plot(pulse_pred)
    #plt.title('Pulse Prediction')
    #plt.subplot(212)
    #plt.plot(resp_pred)
    #plt.title('Resp Prediction')
    #plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    args = parser.parse_args()

    predict_vitals(args)