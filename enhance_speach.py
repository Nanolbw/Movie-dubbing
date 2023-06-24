
import numpy as np

import nextpow2
import math


def speech_enhance(x,fs):

    len_ = 20 * fs // 1000 - 1

    PERC = 50
    len1 = len_ * PERC // 100

    len2 = len_ - len1

    Thres = 0
    Expnt = 2.0
    beta = 0.0000000001
    G = 0.9
    win = np.hamming(len_)

    winGain = len2 / sum(win)

    nFFT = 2 * 2 ** (nextpow2.nextpow2(len_))

    noise_mean = np.zeros(nFFT)

    j = 0
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
        j = j + len_
    noise_mu = noise_mean / 5

    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    for n in range(0, Nframes):

        insign = win * x[k - 1:k + len_ - 1]

        spec = np.fft.fft(insign, nFFT)
        sig = abs(spec)
        theta = np.angle(spec)
        SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        def berouti(SNR):
            if -5.0 <= SNR <= 20.0:
                a = 4 - SNR * 3 / 20
            else:
                if SNR < -5.0:
                    a = 5
                if SNR > 20:
                    a = 1
            return a

        def berouti1(SNR):
            if -5.0 <= SNR <= 20.0:
                a = 3 - SNR * 2 / 20
            else:
                if SNR < -5.0:
                    a = 4
                if SNR > 20:
                    a = 1
            return a

        if Expnt == 1.0:  #
            alpha = berouti1(SNRseg)
        else:
            alpha = berouti(SNRseg)

        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt
        diffw = sub_speech - beta * noise_mu ** Expnt

        def find_index(x_list):
            index_list = []
            for i in range(len(x_list)):
                if x_list[i] < 0:
                    index_list.append(i)
            return index_list

        z = find_index(diffw)
        if len(z) > 0:

            sub_speech[z] = beta * noise_mu[z] ** Expnt

        if SNRseg < Thres:
            noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt
            noise_mu = noise_temp ** (1 / Expnt)
        sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
        x_phase = (sub_speech ** (1 / Expnt)) * (
                    np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))


        xi = np.fft.ifft(x_phase).real

        xfinal[k - 1:k + len2 - 1] = x_old + xi[0:len1]
        x_old = xi[len1:len_]
        k = k + len2

    wave_data = (winGain * xfinal).astype(np.short)
    return wave_data







