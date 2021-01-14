import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator


ckpt_path = 'logs_dir/autovc_one_hot146000.ckpt'
conversion_list_path = 'conversion_list.txt'
data_dir = '../AutoVC_hujk17/full_106_spmel_nosli'
speaker_id_dict_path = '../AutoVC_hujk17/full_106_spmel_nosli/speaker_seen_unseen.txt'

dim_neck = 32
dim_emb = 256
dim_pre = 512
freq = 32
# look up table用, 102个人, 用128作为上限
speaker_num =128



def pad_seq(x, base=freq):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def text2dict(file):
    speaker_id_dict = {}
    f = open(file, 'r').readlines()
    for i, name in enumerate(f):
        name = name.strip().split('|')[0]
        speaker_id_dict[name] = i
    # print(speaker_id_dict)
    return speaker_id_dict


def main():
    # init model
    device = 'cuda:0'
    G = Generator(dim_neck=dim_neck, dim_emb=dim_emb, dim_pre=dim_pre, freq=freq, speaker_num=speaker_num).eval().to(device)
    g_checkpoint = torch.load(ckpt_path)
    G.load_state_dict(g_checkpoint['model'])

    # init speaker name -> id
    speaker_id_dict = text2dict(speaker_id_dict_path)


    # p228/p228_077.npy|p228|p227
    f = open(conversion_list_path, 'r').readlines()
    tasks = [i.strip() for i in f]


    spect_vc = []
    for task in tasks:
        task = task.split('|')
        assert len(task) == 3
        mel_path = task[0]
        s_name = task[1]
        t_name = task[2]

        # process from string -> data: mel, s, t
        mel = np.load(os.path.join(data_dir, mel_path))
        mel, len_pad = pad_seq(mel)
        s_id = speaker_id_dict[s_name]
        t_id = speaker_id_dict[t_name]

        # process from data -> batch tensor: mel, s, t
        mel = torch.from_numpy(mel[np.newaxis, :, :]).to(device)
        s_id = torch.from_numpy(np.asarray([s_id])).to(device)
        t_id = torch.from_numpy(np.asarray([t_id])).to(device)
        print('speaker model out----------', s_id.size())


        with torch.no_grad():
            _, x_identic_psnt, _ = G(mel, s_id, t_id)
            print('mel size:', x_identic_psnt.size())
        
        if len_pad == 0:
            # uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            x_identic_psnt = x_identic_psnt[0, :, :].cpu().numpy()
        else:
            # uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            x_identic_psnt = x_identic_psnt[0, :-len_pad, :].cpu().numpy()
            
        spect_vc.append( ('{}x{}'.format(s_name, t_name), x_identic_psnt) )

    # # speaker = []

    # spect_vc = []

    # for sbmt_i in metadata:
                
    #     x_org = sbmt_i[2]
    #     x_org, len_pad = pad_seq(x_org)
        
    #     uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    #     emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
        
    #     for sbmt_j in metadata:
                    
    #         emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
            
    #         with torch.no_grad():
    #             _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
    #             print('mel size:', x_identic_psnt.size())
                
    #         if len_pad == 0:
    #             # uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
    #             uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
    #         else:
    #             # uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
    #             uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()
            
    #         spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
            
            
    with open('results.pkl', 'wb') as handle:
        pickle.dump(spect_vc, handle)          

if __name__ == "__main__":
    main()