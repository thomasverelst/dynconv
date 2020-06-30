import time
from collections import namedtuple
from string import Template

import cupy
import torch
import math
# HELPER FUNCTIONS

Stream = namedtuple('Stream', ['ptr'])

ROUNDING = 8
CUDA_NUM_THREADS = 256
CHANNELS_PER_THREAD = 2

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code,options=('--restrict','--use_fast_math'))
    return kernel_code.get_function(kernel_name)

def GET_BLOCKS(N, NTHREADS):
    return min((N + NTHREADS - 1) // (NTHREADS), 256*256-1)

def cudaok(x):
    return x.is_cuda and x.is_contiguous()

def ctime():
    torch.cuda.synchronize()
    return time.perf_counter()


# CONV

_masked_conv_kernel = '''
extern "C"
#define CHANNELS ${channels}
#define WIDTH ${width}
#define HEIGHT ${height}
#define OUT_WIDTH ${out_width}
#define OUT_HEIGHT ${out_height}
#define KERNEL_H ${kernel_h}
#define KERNEL_W ${kernel_w}

__global__ void masked_conv(
    const float* const in, const float* const weight_data, float* const out, 
    const int* const active_positions_list, const int* const active_positions_list_inverted, const int nblocks
){
for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < nblocks; i += blockDim.y * gridDim.y){
    const int tile_index = active_positions_list[i];
    const int n = tile_index / (OUT_HEIGHT * OUT_WIDTH);
    const int h = (tile_index / OUT_WIDTH) % OUT_HEIGHT;
    const int w = tile_index % OUT_WIDTH;
    for(int c = threadIdx.x; c < CHANNELS; c += blockDim.x ){        
        float value = 0;
        const float* weight_index = weight_data + c * KERNEL_H * KERNEL_W;
        int tile_val;

        for (int kh = 0; kh < KERNEL_H; ++kh) {
            for (int kw = 0; kw < KERNEL_W; ++kw) {
                const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
                const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
                if ((h_in >= 0) && (h_in < HEIGHT)
                    && (w_in >= 0) && (w_in < WIDTH)) {
                        const int tile_offset =  n*WIDTH*HEIGHT+h_in*WIDTH+w_in;
                        tile_val = active_positions_list_inverted[tile_offset];
                        //if(tile_val >= 0){ // just for safety
                        const int offset =  tile_val*CHANNELS+c;
                        value += (*weight_index) * in[offset];
                        //}
                }
                ++weight_index;
            }
        }
        const int out_offset = i*CHANNELS+c;
        out[out_offset] = value;
    }
}
}
'''

def masked_conv_dw(xsparse, w, active_positions_list, active_positions_list_inverted, xshape, stride, padding, dilation, batch_stride=ROUNDING):
    assert active_positions_list.dtype == torch.int
    assert active_positions_list_inverted.dtype == torch.int
    assert len(xshape) == 4, xshape
    assert w.shape[1] == 1 # dw convolution
    # assert w.shape[2] == 3 # could work with different kernel size but never tested
    # assert w.shape[3] == 3
    assert cudaok(xsparse)
    assert cudaok(active_positions_list_inverted)
    assert cudaok(active_positions_list_inverted)
    assert cudaok(w)
    assert xsparse.shape[2] == 1
    assert xsparse.shape[3] == 1
    
    batch_size, _, height, width = xshape
    channels = xsparse.shape[1]
    kernel_h, kernel_w = w.size()[2:]
    output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
    output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)
    nblocks = len(active_positions_list)

    n2 = (nblocks//batch_stride + 1)*batch_stride if batch_stride > 0 else nblocks
    output =  torch.empty((n2, channels, 1, 1), device='cuda')
    output[nblocks:] = 0

    threadsx =  min(math.ceil(channels/CHANNELS_PER_THREAD), CUDA_NUM_THREADS) 
    threadsy = max(CUDA_NUM_THREADS//threadsx, 1)
    block = (threadsx, threadsy,1)
    grid = (1,GET_BLOCKS(len(active_positions_list), threadsy),1)

    with torch.cuda.device_of(xsparse):
        f = load_kernel('masked_conv', _masked_conv_kernel, 
                        batchsize=batch_size, channels=channels,
                        height=height, width=width,
                        out_height=output_h, out_width=output_w,
                        kernel_h=kernel_h, kernel_w=kernel_w,
                        stride_h=stride[0], stride_w=stride[1],
                        dilation_h=dilation[0], dilation_w=dilation[1],
                        pad_h=padding[0], pad_w=padding[1])
        f(block=block, grid=grid,
            args=[xsparse.data_ptr(), w.data_ptr(), output.data_ptr(), active_positions_list.data_ptr(), active_positions_list_inverted.data_ptr(), int(nblocks)],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return output

## BATCHNORM

_bn_relu_kernel = '''
extern "C"
#define CHANNELS ${channels}

__global__ void bn_relu(
    float* const __restrict__ x, const float* const __restrict__ scale, const float* const __restrict__ bias, const int nblocks
){
for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < nblocks; i += blockDim.y * gridDim.y){
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < CHANNELS; c += blockDim.x * gridDim.x){
        float val = x[i*CHANNELS+c];
        val = val * scale[c] + bias[c];
        if(${relu}){
            val = fmaxf(val, 0); 
        }
        if(${relu6}){ 
            val = fmaxf(val, 0);
            val = fminf(val, 6);
        }
        if(${hswish}){
            float hswishval = val + 3;
            hswishval = fmaxf(hswishval, 0);
            hswishval = fminf(hswishval, 6);
            hswishval = hswishval / 6;
            val = val * hswishval;
        }
        x[i*CHANNELS+c] = val;
    }
}
}
'''

def batchnorm(x, weight, bias, running_mean, running_var, relu=False, relu6=False, hswish=False):
    batchsize, channels, width, height = x.shape
    scale = weight/torch.sqrt(running_var+1e-5)
    bias = bias - scale*running_mean

    threadsx =  min(channels, CUDA_NUM_THREADS) 
    threadsy = max(CUDA_NUM_THREADS//threadsx, 1)
    block = (threadsx, threadsy,1)
    grid = (1,GET_BLOCKS(batchsize, threadsy),1)


    assert width == 1, width
    assert height == 1, height
    assert scale.numel() == channels
    assert bias.numel() == channels
    assert scale.is_cuda
    assert bias.is_cuda

    with torch.cuda.device_of(x):
        f = load_kernel('bn_relu', _bn_relu_kernel, channels=channels,
        relu=int(relu), relu6=int(relu6), hswish=int(hswish))
        f(block=block, grid=grid,
            args=[x.data_ptr(), scale.data_ptr(), bias.data_ptr(), batchsize],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return x

# GATHER - SCATTER

_gather_kernel = '''
extern "C"
__global__ void gather_kernel(const float* __restrict__  const data_in, float* __restrict__ const data_out, const int* __restrict__  const active_positions_list, const int n) {

#define BATCHSIZE ${batchsize}
#define CHANNELS ${channels}
#define WIDTH ${width}
#define HEIGHT ${height}

for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y){
    const int k = (int) active_positions_list[i];
    const int b = k / (WIDTH * HEIGHT);
    const int pos = k % (WIDTH*HEIGHT);
    const int offset = b * CHANNELS * WIDTH * HEIGHT + pos;  
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; \
        c < CHANNELS; c += blockDim.x * gridDim.x){
        data_out[i*CHANNELS+c] = data_in[offset+c*WIDTH*HEIGHT];
    }
}
}
'''

def gather(data_in, m, divisible=ROUNDING):
    active_positions_list = m.active_positions_list
    with torch.cuda.device_of(data_in):
        batchsize, channels, width, height = data_in.shape

        assert cudaok(data_in)
        assert cudaok(active_positions_list)
        npixels = len(active_positions_list)
        n_out = (npixels//divisible + 1)*divisible if divisible > 0 else npixels
        data_out = torch.empty((n_out, channels, 1, 1), device='cuda')
        data_out[npixels:] = 0

        threadsx =  min(math.ceil(channels/CHANNELS_PER_THREAD), CUDA_NUM_THREADS) 
        threadsy = max(CUDA_NUM_THREADS//threadsx, 1)
        block = (threadsx, threadsy,1)
        grid = (1,GET_BLOCKS(npixels, threadsy),1)

        f = load_kernel('gather_kernel', _gather_kernel, 
            batchsize=batchsize, channels=channels,
            width=width, height=height)
        f(block=block, grid=grid,
            args=[data_in.data_ptr(), data_out.data_ptr(), active_positions_list.data_ptr(), int(npixels)],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return data_out


_scatter_kernel = '''
extern "C"
__global__ void scatter_kernel(const float* __restrict__  const data_in, float* __restrict__ const data_out, const int* __restrict__  const active_positions_list, const int n) {

#define BATCHSIZE ${batchsize}
#define CHANNELS ${channels}
#define WIDTH ${width}
#define HEIGHT ${height}
#define OUTPUT_CHANNEL_OFFSET ${output_channel_offset}

for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y){
    const int k = (int) active_positions_list[i];
    const int b = k / (WIDTH * HEIGHT);
    const int pos = k % (WIDTH*HEIGHT);
    const int offset = b * CHANNELS * WIDTH * HEIGHT + pos;  
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; \
        c < CHANNELS; c += blockDim.x * gridDim.x){
        if(${sum_result} > 0){
            atomicAdd(data_out+offset+c*WIDTH*HEIGHT, data_in[i*CHANNELS+c]);          
        } else {
            data_out[offset+(OUTPUT_CHANNEL_OFFSET+c)*WIDTH*HEIGHT] = data_in[i*CHANNELS+c];
        }
    }
}
}
'''

def scatter(data_in, data_out, m, sum_out=False, output_channel_offset=0):
    batchsize, channels, width, height = data_out.shape
    active_positions_list = m.active_positions_list

    assert cudaok(data_in)
    assert cudaok(data_out)
    assert cudaok(active_positions_list)
    assert len(active_positions_list) <= data_in.shape[0], (len(active_positions_list), data_in.shape)
    assert data_in.shape[1] == (data_out.shape[1] - output_channel_offset)
    assert data_in.shape[2] == 1
    assert data_in.shape[3] == 1

    threadsx =  min(math.ceil(channels/CHANNELS_PER_THREAD), CUDA_NUM_THREADS) 
    threadsy = max(CUDA_NUM_THREADS//threadsx, 1)
    block = (threadsx, threadsy,1)
    grid = (1,GET_BLOCKS(len(active_positions_list), threadsy),1)

    with torch.cuda.device_of(data_in):
        f = load_kernel('scatter_kernel', _scatter_kernel, 
            batchsize=batchsize, channels=channels,
            width=width, height=height, sum_result=int(sum_out), output_channel_offset=output_channel_offset)
        f(block=block, grid=grid,
            args=[data_in.data_ptr(), data_out.data_ptr(), active_positions_list.data_ptr(), int(len(active_positions_list))],
            stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return data_out
