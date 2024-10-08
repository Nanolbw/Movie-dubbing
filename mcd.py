import librosa
import math
import numpy as np
import pyworld
import pysptk
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# ================================================= #
# calculate the Mel-Cepstral Distortion (MCD) value #
# ================================================= #
# class Calculate_MCD(object):
#     """docstring for Calculate_MCD"""

#     def __init__(self, MCD_mode):
#         super(Calculate_MCD, self).__init__()
#         # self.args = args
#         self.MCD_mode = MCD_mode
#         self.SAMPLING_RATE = 22050
#         self.FRAME_PERIOD = 5.0
#         self.log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)  # 6.141851463713754
#         self.min_cost_tot = 0.0
#         self.frames_tot = 0
#         self.mean_mcd = 0.0

#     def load_wav(self, wav_file, sample_rate):
#         """
#         Load a wav file with librosa.
#         :param wav_file: path to wav file
#         :param sr: sampling rate
#         :return: audio time series numpy array
#         """
#         wav, _ = librosa.load(wav_file, sr=sample_rate, mono=True)
#         return wav

#     # distance metric
#     def log_spec_dB_dist(self, x, y):
#         # log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
#         diff = x - y
#         return self.log_spec_dB_const * math.sqrt(np.inner(diff, diff))

#     # calculate distance (metric)
#     # def calculate_mcd_distance(self, x, y, distance, path):
#     def calculate_mcd_distance(self, x, y, path):
#         '''
#         param path: pairs between x and y
#         '''
#         # distance /= (len(x) + len(y))
#         pathx = list(map(lambda l: l[0], path))
#         pathy = list(map(lambda l: l[1], path))
#         x, y = x[pathx], y[pathy]
#         frames = x.shape[0]  # length of pairs

#         # frames = len(path)      # length of pairs
#         # frames = len(x)         # length of reference audios, i.e., ref_mcep_vec
#         self.frames_tot += frames
#         # self.min_cost_tot = distance

#         z = x - y
#         self.min_cost_tot += np.sqrt((z * z).sum(-1)).sum()

#     # extract acoustic features
#     # alpha = 0.65  # commonly used at 22050 Hz
#     # def wav2mcep_numpy(self, wavfile, target_directory, alpha=0.65, fft_size=512, mcep_size=34):
#     # 	# make relevant directories
#     # 	if not os.path.exists(target_directory):
#     # 		os.makedirs(target_directory)
#     # def wav2mcep_numpy(self, wavfile, alpha=0.65, fft_size=512, mcep_size=34):
#     def wav2mcep_numpy(self, loaded_wav, alpha=0.65, fft_size=512):
#         # # load wav from given wavfile
#         # loaded_wav = self.load_wav(wavfile, sample_rate=self.SAMPLING_RATE)

#         # Use WORLD vocoder to spectral envelope
#         _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=self.SAMPLING_RATE,
#                                      frame_period=self.FRAME_PERIOD, fft_size=fft_size)

#         # Extract MCEP features
#         mcep = pysptk.sptk.mcep(sp, order=13, alpha=alpha, maxiter=0,
#                                 etype=1, eps=1.0E-8, min_det=0.0, itype=3)

#         # # Extract MFCC features
#         # mfcc = pysptk.sptk.mfcc(sp, order=self.args.order, fs=self.SAMPLING_RATE, eps=1.0E-8)

#         # fname = os.path.basename(wavfile).split('.')[0]
#         # np.save(os.path.join(target_directory, fname + '.npy'),
#         # 		mgc,
#         # 		allow_pickle=False)

#         return mcep

#     # calculate the Mel-Cepstral Distortion (MCD) value
#     # def average_mcd(self, ref_mcep_files, synth_mcep_files, cost_function):
#     def average_mcd(self, ref_audio_file, syn_audio_file, cost_function, MCD_mode, cofs, cofs_batch=None):
#         """
#         Calculate the average MCD.
#         :param ref_mcep_files: list of strings, paths to MCEP target reference files
#         :param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
#         :param cost_function: distance metric used
#         :param plain: if plain=True, use Dynamic Time Warping (dtw)
#         :returns: average MCD, total frames processed
#         """
#         # load wav from given wav file
#         loaded_ref_wav = self.load_wav(ref_audio_file, sample_rate=self.SAMPLING_RATE)
#         loaded_syn_wav = self.load_wav(syn_audio_file, sample_rate=self.SAMPLING_RATE)

#         if MCD_mode == "plain":
#             # pad 0
#             if len(loaded_ref_wav) < len(loaded_syn_wav):
#                 loaded_ref_wav = np.pad(loaded_ref_wav, (0, len(loaded_syn_wav) - len(loaded_ref_wav)))
#             else:
#                 loaded_syn_wav = np.pad(loaded_syn_wav, (0, len(loaded_ref_wav) - len(loaded_syn_wav)))

#         # extract MCEP features (vectors): 2D matrix (num x mcep_size)
#         # ref_mcep_vec = self.wav2mcep_numpy(ref_audio_file)
#         # syn_mcep_vec = self.wav2mcep_numpy(syn_audio_file)
#         ref_mcep_vec = self.wav2mcep_numpy(loaded_ref_wav)
#         syn_mcep_vec = self.wav2mcep_numpy(loaded_syn_wav)

#         if MCD_mode == "plain":
#             # print("Calculate plain MCD ...")
#             # num_temp = len(ref_mcep_vec) if len(ref_mcep_vec)>len(syn_mcep_vec) else len(syn_mcep_vec)
#             path = []
#             # for i in range(num_temp):
#             for i in range(len(ref_mcep_vec)):
#                 path.append((i, i))
#         # # pad 0
#         # ref_mcep_vec = np.pad(ref_mcep_vec, ((0, num_temp-ref_mcep_vec.shape[0]), (0, 0)))
#         # syn_mcep_vec = np.pad(syn_mcep_vec, ((0, num_temp-syn_mcep_vec.shape[0]), (0, 0)))
#         elif MCD_mode == "dtw":
#             # dynamic time warping using librosa
#             # # param metric: (str, e.g. "euclidean") Identifier for the cost-function as documented in scipy.spatial.distance.cdist()
#             # min_cost, _ = librosa.sequence.dtw(ref_mcep_vec[:, 1:].T, syn_mcep_vec[:, 1:].T,
#             # 								   metric="euclidean")
#             # distance, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)
#             # print("Calculate MCD-dtw ...")
#             _, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)
#         elif MCD_mode == "adv_dtw":
#             epsilon = 1e-5
#             # cof = len(ref_mcep_vec)/len(syn_mcep_vec) if len(ref_mcep_vec)>len(syn_mcep_vec) else len(syn_mcep_vec)/len(ref_mcep_vec)
#             cof = cofs[0] / (cofs[1] + epsilon) if cofs[0] > cofs[1] else cofs[1] / (cofs[0] + epsilon)
#             _, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)

#         # self.calculate_mcd_distance(ref_mcep_vec, syn_mcep_vec, distance, path)
#         self.calculate_mcd_distance(ref_mcep_vec, syn_mcep_vec, path)

#         if MCD_mode == "adv_dtw":
#             self.mean_mcd += cof * self.log_spec_dB_const * self.min_cost_tot / self.frames_tot
#         else:
#             self.mean_mcd += self.log_spec_dB_const * self.min_cost_tot / self.frames_tot

#         # reset self.min_cost_tot and self.frames_tot
#         self.min_cost_tot = 0.0
#         self.frames_tot = 0

#     # calculate mcd
#     def calculate_mcd(self, reference_audio, synthesized_audio, num_audio, cofs=None, average=False):
#         # reference (target) audio and synthesized wav (the filepath contains both path and file name)
#         # reference_audio = os.path.join(args.datasets_root, args.current_dataset_name,
#         # 	args.current_speaker_name, args.current_utterance_name)
#         # synthesized_audio = args.synthesized_wav_name

#         # synthesized_audio = os.path.join(args.datasets_root, args.current_dataset_name,
#         # 	args.current_speaker_name, args.current_utterance_name)

#         # extract acoustic features
#         self.average_mcd(reference_audio, synthesized_audio, self.log_spec_dB_dist, self.MCD_mode, cofs)
#         if average:
#             avg_mcd = self.mean_mcd / num_audio
#             self.mean_mcd = 0.0  # clean mean_mcd
#             # print("MCD = {}, calculated over a total of {} frames".format(mcd_value, total_frames_used))
#             # print("MCD = {}, calculated over a total of {} audios".format(avg_mcd, num_audio))
#             # return "MCD = {}, calculated over a total of {} audios".format(avg_mcd, num_audio)
#             return avg_mcd

class Calculate_MCD(object):
	"""docstring for Calculate_MCD"""
	def __init__(self, MCD_mode):
		super(Calculate_MCD, self).__init__()
		# self.args = args
		self.MCD_mode = MCD_mode
		self.SAMPLING_RATE = 22050
		self.FRAME_PERIOD = 5.0
		self.log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0) # 6.141851463713754
		self.min_cost_tot = 0.0
		self.frames_tot = 0
		self.mean_mcd = 0.0
	
	def load_wav(self, wav_file, sample_rate):
		"""
		Load a wav file with librosa.
		:param wav_file: path to wav file
		:param sr: sampling rate
		:return: audio time series numpy array
		"""
		wav, _ = librosa.load(wav_file, sr=sample_rate, mono=True)
		return wav

	# distance metric
	def log_spec_dB_dist(self, x, y):
		# log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
		diff = x - y
		return self.log_spec_dB_const * math.sqrt(np.inner(diff, diff))

	# calculate distance (metric)
	# def calculate_mcd_distance(self, x, y, distance, path):
	def calculate_mcd_distance(self, x, y, path):
		'''
		param path: pairs between x and y
		'''
		pathx = list(map(lambda l: l[0], path))
		pathy = list(map(lambda l: l[1], path))
		x, y = x[pathx], y[pathy]
		self.frames_tot += x.shape[0]       # length of pairs

		z = x - y
		# min_cost_tot = np.sqrt((z * z).sum(-1)).sum()
		self.min_cost_tot += np.sqrt((z * z).sum(-1)).sum()
		

	# extract acoustic features
	# alpha = 0.65  # commonly used at 22050 Hz
	def wav2mcep_numpy(self, loaded_wav, alpha=0.65, fft_size=512):

		# Use WORLD vocoder to spectral envelope
		_, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=self.SAMPLING_RATE,
									   frame_period=self.FRAME_PERIOD, fft_size=fft_size)

		# Extract MCEP features
		mcep = pysptk.sptk.mcep(sp, order=13, alpha=alpha, maxiter=0,
							   etype=1, eps=1.0E-8, min_det=0.0, itype=3)

		return mcep

	# calculate the Mel-Cepstral Distortion (MCD) value
	def average_mcd(self, ref_audio_file, syn_audio_file, cost_function, MCD_mode):
		"""
		Calculate the average MCD.
		:param ref_mcep_files: list of strings, paths to MCEP target reference files
		:param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
		:param cost_function: distance metric used
		:param plain: if plain=True, use Dynamic Time Warping (dtw)
		:returns: average MCD, total frames processed
		"""
		# load wav from given wav file
		loaded_ref_wav = self.load_wav(ref_audio_file, sample_rate=self.SAMPLING_RATE)
		loaded_syn_wav = self.load_wav(syn_audio_file, sample_rate=self.SAMPLING_RATE)

		if MCD_mode == "plain":
			# pad 0
			if len(loaded_ref_wav)<len(loaded_syn_wav):
				loaded_ref_wav = np.pad(loaded_ref_wav, (0, len(loaded_syn_wav)-len(loaded_ref_wav)))
			else:
				loaded_syn_wav = np.pad(loaded_syn_wav, (0, len(loaded_ref_wav)-len(loaded_syn_wav)))

		# extract MCEP features (vectors): 2D matrix (num x mcep_size)
		ref_mcep_vec = self.wav2mcep_numpy(loaded_ref_wav)
		syn_mcep_vec = self.wav2mcep_numpy(loaded_syn_wav)

		if MCD_mode == "plain":
			# print("Calculate plain MCD ...")
			path = []
			# for i in range(num_temp):
			for i in range(len(ref_mcep_vec)):
				path.append((i, i))
		elif MCD_mode == "dtw":
			# print("Calculate MCD-dtw ...")
			_, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)
		elif MCD_mode == "adv_dtw":
			# print("Calculate MCD-dtw-sl ...")
			cof = len(ref_mcep_vec)/len(syn_mcep_vec) if len(ref_mcep_vec)>len(syn_mcep_vec) else len(syn_mcep_vec)/len(ref_mcep_vec)
			_, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)

		self.calculate_mcd_distance(ref_mcep_vec, syn_mcep_vec, path)

		if MCD_mode == "adv_dtw":
			self.mean_mcd += cof * self.log_spec_dB_const * self.min_cost_tot / self.frames_tot
		else:
			self.mean_mcd += self.log_spec_dB_const * self.min_cost_tot / self.frames_tot

		self.min_cost_tot = 0.0
		self.frames_tot = 0


	# calculate mcd
	def calculate_mcd(self, reference_audio, synthesized_audio, num_audio=None, average=False):
		# extract acoustic features
		# mean_mcd = self.average_mcd(reference_audio, synthesized_audio, self.log_spec_dB_dist, self.MCD_mode)

		# return mean_mcd
		self.average_mcd(reference_audio, synthesized_audio, self.log_spec_dB_dist, self.MCD_mode)
		if average:
			avg_mcd = self.mean_mcd / num_audio
			self.mean_mcd = 0.0  # clean mean_mcd
			# print("MCD = {}, calculated over a total of {} frames".format(mcd_value, total_frames_used))
			# print("MCD = {}, calculated over a total of {} audios".format(avg_mcd, num_audio))
			# return "MCD = {}, calculated over a total of {} audios".format(avg_mcd, num_audio)
			return avg_mcd