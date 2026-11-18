from candidate import ROOT,build_candidate
ARM='E2.1'
RUN=ROOT/'log/TinyViM_E2.1_内部256_六支残差_HAM_20261105'
CHOICES=dict(
 placement='Align3 512x8x8 -> new E2.1 module -> retained original D3 DoubleConv',
 internal_width=256, external_width=512, band_channels=[86,85,85],
 low='three independent four-direction FullScan Mamba branches',
 high='RepDW3, RepDW7, dilated DW3 dilation2',
 residuals='all six F(x)+alpha*x; one trainable alpha per band initialized1; both outer mixer/FFN residuals retained',
 frequency_split='unchanged complementary ideal FFT radius .25 cycles per pixel',
 gates='only two simultaneous cross-band gates; no intra-band cyclic gates',
 attention='none',ffn='256->1024->256',
 pre_post='depthwise3 + pointwise1 + BN + GELU; 512->256 / 256->512',
 fusion='concat512 -> linear DW3(512) -> PW1(512,256)',
 initialization='official ImageNet encoder; original E0 decoder42 and align43; independent added module42',
 comparison='E0-based independent HAM experiment; no trained HAM weights loaded',
)
