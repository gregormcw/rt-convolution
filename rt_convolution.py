import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from IPython.display import Audio

x, fs_1 = sf.read("snare.wav")
h, fs_2 = sf.read("gmcwilliam_ass3_ir.wav")
x = x[:,0]
h = h[:,0]

def d_conv(x, h):

    # Ensure both x and h are type ndarray
    x = np.array(x)
    h = np.array(h)

    # Set variable lengths to appropriate values
    N = len(x)
    K = len(h)

    # Create ndarray of zeros, of length N+K-1
    y = np.zeros(N + K - 1)

    # Calculate the convolution
    for k in range(K):
        y[k:N + k] += h[k] * x

    return y


def f_conv(x, h):

    # Prepare zeros so that signal length can be matched
    x_zeros = np.zeros(len(h) - 1)

    # Append x to end of zeros
    x_zeropad = np.hstack([x, x_zeros])

    # Prepare zeros so that signal length can be matched
    h_zeros = np.zeros(len(x) - 1)

    # Append h to end of zeros
    h_zeropad = np.hstack([h, h_zeros])

    # Compute FFT of zero-padded input
    x_fft = np.fft.fft(x_zeropad)

    # Compute FFT of zero-padded convolutional filter
    h_fft = np.fft.fft(h_zeropad)

    # Compute IFFT of f-d multiplied (and hence t-d convolved) x and h
    x_conv = np.real(np.fft.ifft(x_fft * h_fft))

    # Return the time-domain convolution of x and h
    return x_conv


# Main algorithm

def convolver(h, x, conv_method, frame_length):
    # INPUTS
    # ir:  array containing the impulse response
    # x: array containing a signal
    # conv_method: convolution method:
    #               1: direct convolution
    #               2: fast convolution
    # frame_length: length of input segments to be convolved with impulse response

    # OUTPUTS:
    # out_x:         Post-convolution output signal vector

    # Ensure both x and h are type ndarray
    x = np.array(x)
    h = np.array(h)

    # Set values for the lengths of x and h
    N = len(x)
    K = len(h)

    # Create an array containing zeros upon which the convolutions can be calculated
    out_x = np.zeros(N + K - 1)

    # Set the starting index to 0
    n = 0

    err_msg = "Invalid conv_method input. Enter 1 for direct convolution, 2 for fast convolution."

    # Allow user to select desired convolution function
    if conv_method == 1:
        # Bind the selected convolution function to a new variable to avoid repeating code
        sel_conv = d_conv
    elif conv_method == 2:
        sel_conv = f_conv
    else:
        # Print error message if an invalid input is entered for convolution type
        print(err_msg)
        return

    while n < N:

        # Selected convolution function performed on current segment
        cur_conv = sel_conv(x[n:frame_length + n], h)

        # Current segment added to existing output signal following convolution, from point n
        out_x[n:len(
            cur_conv) + n] += cur_conv

        # n hops forward by segment length
        n += frame_length

    # Return the convolved output signal
    return out_x

# Create list of convolution outputs to avoid repeating code
convs = [conv_1, conv_2]

for conv in convs:
    #if max(abs(conv)) > 1: # Uncomment if you wish only to normalize if clipping is occuring
    # Normalize the signal
    conv *= 1/max(abs(conv))