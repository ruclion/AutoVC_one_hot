import torch
import pickle
import librosa
import numpy as np
from synthesis import wavegen
from synthesis import build_model



device = torch.device("cuda")
model_path = "pretrained-checkpoint_step001000000_ema.pth"



def load_model():
    model = build_model().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def vocode_spec(spec, model, out_name):
    c = spec
    waveform = wavegen(model, c=c)   
    librosa.output.write_wav(out_name, waveform, sr=16000)


if __name__ == "__main__":
    demo_spec_by_auto_vc_path = '/ceph/home/hujk17/AutoVC_hujk17/full_106_spmel_nosli/p225/p225_003.npy'
    name = 'test_mel_rec_auto_vc_WaveNet.wav'

    model = load_model()
    print('model loaded finish...')

    spec = np.load(demo_spec_by_auto_vc_path)
    vocode_spec(spec, model, name)

    
