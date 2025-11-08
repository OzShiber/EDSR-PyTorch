from model import common

import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # --- START: Modified body for GSE Ablation ---
        m_body = []
        for i in range(n_resblocks):
            # 1. Create the standard ResBlock
            resblock = common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            )

            # 2. Check if we should wrap it with GSE
            if args.use_gse and (i % args.gse_freq == 0):
                # YES. Wrap it and pass the controls.
                gse_wrapper = common.GSEWrapper(
                    resblock, 
                    median=args.gse_median, 
                    scale_factor=args.gse_scale_factor
                )
                m_body.append(gse_wrapper)
            else:
                # NO. Just add the standard ResBlock.
                m_body.append(resblock)

        # 3. Create the final conv layer
        last_conv = conv(n_feats, n_feats, kernel_size)

        # 4. Conditionally wrap the *last* layer
        # We check if the *index* of this layer (n_resblocks) matches the freq
        if args.use_gse and (n_resblocks % args.gse_freq == 0):
             gse_wrapper = common.GSEWrapper(
                last_conv, 
                median=args.gse_median, 
                scale_factor=args.gse_scale_factor
            )
             m_body.append(gse_wrapper)
        else:
            m_body.append(last_conv)
        # --- END: Modified body for GSE Ablation ---

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key \"{}\" in state_dict'
                                   .format(name))