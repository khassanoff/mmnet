# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchaudio
import glob
import re
import copy
import json
import random
import editdistance
import librosa
import csv, pdb


class MyDataset(Dataset):
    def __init__(self, opt, data_type='train', noise=False):

        #pdb_bool = True

        if data_type.lower() == 'train':
            self.data_path = opt.train_path
        elif data_type.lower() == 'valid':
            self.data_path = opt.valid_path
        elif data_type.lower() == 'test':
            self.data_path = opt.test_path

        self.opt = opt
        self.noise = noise
        if self.noise:
            if opt.mode in [1,4,5,7] and opt.add_rgb_noise:       #rgb images
                print("Adding {} noise to visual data (STD {})".format(opt.rgb_noise, opt.rnoise_value))
            if opt.mode in [2,4,6,7] and opt.add_thr_noise:       #rgb images
                print("Adding {} noise to thermal data (STD {})".format(opt.thr_noise, opt.tnoise_value))
            if opt.mode in [3,5,6,7] and opt.add_audio_noise:       #rgb images
                print("Adding {} noise to audio data (STD {})".format(opt.audio_noise, opt.anoise_value))

        self.sub_path       = opt.sub_path
        self.num_frames     = opt.num_frames
        self.mode           = opt.mode
        self.sr             = opt.sample_rate
        self.segment_len    = opt.segment_len
 
        self.sub_label  = {}
        with open(self.sub_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if opt.predict.lower() == "age":
                    if int(row[2]) <= 24:
                        self.sub_label[row[0]] = 0
                    elif int(row[2]) <= 34:
                        self.sub_label[row[0]] = 1
                    else:
                        self.sub_label[row[0]] = 2
                elif opt.predict.lower() == "gender":
                    if row[3].lower() == 'female':
                        self.sub_label[row[0]] = float(0)
                    else:
                        self.sub_label[row[0]] = float(1)
                else:
                    print("Incorrect prediction value '{}' is given! TERMINATING...")
                    exit()

        subjects = os.listdir(self.data_path)
        subjects = list(filter(lambda sub: sub.find('sub_') != -1, subjects))
        subjects = sorted(subjects, key=lambda sub: int(sub.split('_')[1]))

        self.data = []
        for sub in subjects:
            label = self.sub_label[sub.split('_')[1]]

            trials = os.listdir(os.path.join(self.data_path,sub))
            trials = list(filter(lambda trial: trial.find('trial_') != -1, trials))
            trials = sorted(trials, key=lambda trial: int(trial.split('_')[1]))
            assert len(trials) == 2, "Incorrect number of trials: "+sub
            for trial in trials:
                records = os.listdir(os.path.join(self.data_path,sub,trial,"mic1_audio_cmd_trim"))
                records = list(filter(lambda record: record.find('.wav') != -1, records))
                records = sorted(records, key=lambda record: int(record.split('_')[4]))

                cmds = [record.split('_')[4] for record in records]

                if self.mode in [1,4,5,7]:       #rgb images
                    if not os.path.exists(os.path.join(self.data_path,sub,trial,"rgb_image_cmd_aligned")):
                        print(os.path.join(self.data_path,sub,trial,"rgb_image_cmd_aligned"))
                        continue
                    rgb_images = os.listdir(os.path.join(self.data_path,sub,trial,"rgb_image_cmd_aligned"))
                    rgb_images = list(filter(lambda image: image.find('_3.png') != -1, rgb_images))

                if self.mode in [2,4,6,7]:     #thermal images
                    if not os.path.exists(os.path.join(self.data_path,sub,trial,"thr_image_cmd")):
                        print(os.path.join(self.data_path,sub,trial,"thr_image_cmd"))
                        continue
                    thr_images = os.listdir(os.path.join(self.data_path,sub,trial,"thr_image_cmd"))
                    thr_images = list(filter(lambda image: image.find('_1.png') != -1, thr_images))

                for record in records:
                    cmd = record.split('_')[4]

                    if self.mode in [1,4,5,7]:       #rgb images
                        rgb_images_tmp = list(filter(lambda image: image.split('_')[4] == cmd, rgb_images))
                        rgb_images_tmp = sorted(rgb_images_tmp, key=lambda image: int(image.split('_')[5]))
                        if len(rgb_images_tmp) < self.num_frames:
                            print("This record has insufficient number of rgb frames: "+record)
                            continue
                        #downsample image stream
                        rgb_images_tmp = [rgb_images_tmp[i] for i in np.linspace(0,len(rgb_images_tmp)-1,
                                            endpoint=True,num=self.num_frames,dtype=int).tolist()]
                        #read images into an array list
                        rgb_array = [cv2.imread(os.path.join(self.data_path,sub,trial,
                                        "rgb_image_cmd_aligned",image)) for image in rgb_images_tmp]
                        rgb_array = list(filter(lambda image: not image is None, rgb_array))

                        #add noise
                        if self.noise and opt.add_rgb_noise:
                            rgb_array = [self.add_image_noise(image, opt.rgb_noise, opt.rnoise_value)
                                            for image in rgb_array]

                        #reduce image dimension
                        rgb_array = [cv2.resize(image, (116, 87), interpolation=cv2.INTER_LANCZOS4)
                                        for image in rgb_array]
                        #convert array list into numpy
                        rgb_array = np.stack(rgb_array, axis=0).astype(np.float32)

                    if self.mode in [2,4,6,7]:     #thermal images
                        thr_images_tmp = list(filter(lambda image: image.split('_')[4] == cmd, thr_images))
                        thr_images_tmp = sorted(thr_images_tmp, key=lambda image: int(image.split('_')[5]))
                        if len(thr_images_tmp) < self.num_frames:
                            print("This record has insufficient number of thermal frames: "+record)
                            continue
                        #downsample image stream
                        thr_images_tmp = [thr_images_tmp[i] for i in np.linspace(0,len(thr_images_tmp)-1,
                                            endpoint=True,num=self.num_frames,dtype=int).tolist()]
                        #read images into an array list
                        thr_array = [cv2.imread(os.path.join(self.data_path,sub,trial,
                                        "thr_image_cmd",image)) for image in thr_images_tmp]
                        thr_array = list(filter(lambda image: not image is None, thr_array))

                        #add noise
                        if self.noise and opt.add_thr_noise:
                            thr_array = [self.add_image_noise(image, opt.thr_noise, opt.tnoise_value)
                                            for image in thr_array]

                        #reduce image dimension
                        thr_array = [cv2.resize(image, (116, 87), interpolation=cv2.INTER_LANCZOS4)
                                        for image in thr_array]
                        #convert array list into numpy
                        thr_array = np.stack(thr_array, axis=0).astype(np.float32)

                    if self.mode in [3,5,6,7]:
                        #audio, _ = librosa.core.load(os.path.join(self.data_path, sub, trial,
                        #                    'mic1_audio_cmd_trim', record), sr=self.sr)
                        #audio, _ = librosa.effects.trim(audio)
                        audio, samplerate = torchaudio.load(os.path.join(self.data_path, sub,
                                                        trial, 'mic1_audio_cmd_trim', record))
                        audio = audio.squeeze()
                        if self.noise and opt.add_audio_noise:
                            #add additive white Gaussian noise (AWGN)
                            audio    = self.add_audio_noise(audio, opt.audio_noise, opt.anoise_value)
                        if len(audio) < self.segment_len*self.sr:
                            print("This record has insufficient number of audio features: "+record)
                            continue
                        audio = audio[len(audio)//2-int(self.segment_len*self.sr)//2:
                                      len(audio)//2+int(self.segment_len*self.sr)//2]

                        if False:
                        # spectogram features
                            spec = np.abs(librosa.stft(audio)) # energy
                            #Mean-Var normalization
                            mu = np.mean(spec, 0, keepdims=True)
                            std = np.std(spec, 0, keepdims=True)
                            spec = (spec - mu)/(std + 1e-5)
                        else:
                        # mel spectogram features
                            #spec = librosa.feature.melspectrogram(audio, sr=self.sr)
                            spec = torchaudio.transforms.MelSpectrogram()(audio)
                            ##spec = np.stack(spec, axis=0).astype(np.float32)
                            spec = spec.transpose(1,0) # (Feature, Time) -> (Time, Feature)
                            spec = spec.unsqueeze(2)
                            spec = spec.unsqueeze(3)
                            #spec = np.expand_dims(spec,axis=(2,3)) # (T,F) -> (T,H,W,C)

                    if self.mode == 1:      #rgb images
                        self.data.append([rgb_array, label, sub])
                    elif self.mode == 2:    #thermal images
                        self.data.append([thr_array, label, sub])
                    elif self.mode == 3:    #audio
                        self.data.append([spec, label, sub])
                    elif self.mode == 4:    #rgb and thermal
                        self.data.append([rgb_array, thr_array, label, sub])
                    elif self.mode == 5:    #rgb and thermal
                        self.data.append([rgb_array, spec, label, sub])
                    elif self.mode == 6:    #thermal and audio
                        self.data.append([thr_array, spec, label, sub])
                    elif self.mode == 7:    #rgb, thermal and audio
                        self.data.append([rgb_array, thr_array, spec, label, sub])
        #print("Total number of recordings: "+str(len(self.data)))


    def add_image_noise(self, image, noise_type='gauss', noise_value=10):
        if noise_type.lower() == "gauss":
            #target_snr_db   = 10*np.log10(noise_value)
            #sig_avg_watts   = np.sum(image**2)/len(image)
            #sig_avg_db      = 10*np.log10(sig_avg_watts)
            #noise_avg_db    = sig_avg_db - target_snr_db
            #noise_avg_watts = 10**(noise_avg_db/10)
            row,col,ch = image.shape
            mean = 0
            #sigma = np.sqrt(noise_avg_watts)
            sigma = noise_value
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy_image = image + gauss
        elif noise_type.lower() == "blur":
            kernel_size = (noise_value, noise_value)
            noisy_image = cv2.blur(image, kernel_size)
        else:
            print("Incorrect image noise type is given! Terminating ...")
            exit()

        return noisy_image.astype("uint8")


    def add_audio_noise(self, audio, noise_type='gauss', noise_value=10):
        if noise_type.lower() == "gauss":
            #target_snr_db   = 10*np.log10(noise_value)
            # Calculate signal power and convert to dB 
            #sig_avg_watts   = np.sum(audio**2)/len(audio)
            #sig_avg_db      = 10*np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            #noise_avg_db    = sig_avg_db - target_snr_db
            #noise_avg_watts = 10**(noise_avg_db/10)
            # Generate an sample of white noise
            mean_noise = 0
            sigma = noise_value
            #sigma = np.sqrt(noise_avg_watts)
            #noise_volts = np.random.normal(mean_noise, sigma, len(audio))
            noise_volts = torch.normal(mean_noise, sigma, size=(len(audio),))
            #noise_volts = noise_volts.astype('float16')
            # print("noise_volts len: "+str(len(noise_volts)))
            # Noise up the original signal
            y_volts = audio + noise_volts
            #y_volts = audio.astype('float16') + noise_volts.astype('float16')
        else:
            print("Incorrect audio noise type is given! Terminating ...")
            exit()

        #return y_volts.astype('float32')
        return y_volts


    def __getitem__(self, idx):
        if self.mode == 1:       #rgb images
            (rgb_images, label, sub) = self.data[idx]
            # (T, H, W, C)->(C, T, H, W)
            return {'rgb_images': torch.FloatTensor(rgb_images.transpose(3, 0, 1, 2)), 
                    'label': label,
                    'sub': sub}
        elif self.mode == 2:     #thermal images
            (thr_images, label, sub) = self.data[idx]
            # (T, H, W, C)->(C, T, H, W)
            return {'thr_images': torch.FloatTensor(thr_images.transpose(3, 0, 1, 2)),
                    'label': label,
                    'sub': sub}
        elif self.mode == 3:     #audio
            (audio, label, sub) = self.data[idx]
            #audio = np.pad(audio,((0,0),(0,0),(0,0),(1,1)))
            audio = F.pad(audio,(1,1),"constant", 0)
            # (T, H, W, C)->(C, T, H, W)
            return {'audio': audio.permute(3, 0, 1, 2),
                    'label': label,
                    'sub': sub}
            #return {'audio': torch.FloatTensor(audio.transpose(3, 0, 1, 2)),
            #        'gender': float(gender),
            #        'sub': sub}
        elif self.mode == 4:       #rgb images
            (rgb_images, thr_images, label, sub) = self.data[idx]
            # (T, H, W, C)->(C, T, H, W)
            return {'rgb_images': torch.FloatTensor(rgb_images.transpose(3, 0, 1, 2)), 
                    'thr_images': torch.FloatTensor(thr_images.transpose(3, 0, 1, 2)),
                    'label': label,
                    'sub': sub}
        elif self.mode == 5:
            (rgb_images, audio, label, sub) = self.data[idx]
            #audio = np.pad(audio,((0,0),(0,0),(0,0),(1,1)))
            audio = F.pad(audio,(1,1),"constant", 0)
            # (T, H, W, C)->(C, T, H, W)
            return {'rgb_images': torch.FloatTensor(rgb_images.transpose(3, 0, 1, 2)), 
                    'audio': audio.permute(3, 0, 1, 2),
                    'label': label,
                    'sub': sub}
        elif self.mode == 6:
            (thr_images, audio, label, sub) = self.data[idx]
            #audio = np.pad(audio,((0,0),(0,0),(0,0),(1,1)))
            audio = F.pad(audio,(1,1),"constant", 0)
            # (T, H, W, C)->(C, T, H, W)
            return {'thr_images': torch.FloatTensor(thr_images.transpose(3, 0, 1, 2)),
                    'audio': audio.permute(3, 0, 1, 2),
                    'label': label,
                    'sub': sub}
        elif self.mode == 7:
            (rgb_images, thr_images, audio, label, sub) = self.data[idx]
            #audio = np.pad(audio,((0,0),(0,0),(0,0),(1,1)))
            audio = F.pad(audio,(1,1),"constant", 0)
            # (T, H, W, C)->(C, T, H, W)
            return {'rgb_images': torch.FloatTensor(rgb_images.transpose(3, 0, 1, 2)), 
                    'thr_images': torch.FloatTensor(thr_images.transpose(3, 0, 1, 2)),
                    #'audio': torch.FloatTensor(audio.transpose(3, 0, 1, 2)),
                    'audio': audio.permute(3, 0, 1, 2),
                    'label': label,
                    'sub': sub}

    def __len__(self):
        return len(self.data)
