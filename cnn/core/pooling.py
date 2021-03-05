import numpy as np


def calc_output_shape(input_shape, pool_size, stride):

    (input_height, input_width) = input_shape
    (pool_height, pool_width) = pool_size
    (stride_y, stride_x) = stride

    output_shape = ((input_height // stride_y), (input_width // stride_x))

    return output_shape


def forward(input, pool_size, stride):

    (batches, input_height, input_width) = input.shape
    (pool_height, pool_width) = pool_size
    (stride_y, stride_x) = stride

    output_shape = calc_output_shape(input.shape[-2:], pool_size, stride)

    output = np.zeros((batches, ) + output_shape)

    input_y = out_y = 0
    while (input_y + pool_height) <= input_height:
        input_x = out_x = 0
        while (input_x + pool_width) <= input_width:
            i = input[:, input_y:input_y + pool_height, input_x:input_x + pool_width]
            max_pool = np.max(i.reshape((batches, -1)), axis=-1)
            output[:, out_y, out_x] = max_pool

            input_x += stride_x
            out_x += 1

        input_y += stride_y
        out_y += 1

    return output


def unravel_indices(input):

    indices = np.nanargmax(input)

    indices = np.unravel_index(indices, input.shape)

    return indices


def backward(input, error, pool_size, pool_stride):

    (batches, input_height, input_width) = input.shape
    (pool_height, pool_width) = pool_size
    (stride_y, stride_x) = pool_stride

    back_layer_error = np.zeros(input.shape)

    input_y = out_y = 0
    while (input_y + pool_height) <= input_height:
        input_x = out_x = 0
        while (input_x + pool_width) <= input_width:

            for b in range(batches):
                (unravel_y, unravel_x) = unravel_indices(input[b, input_y:input_y + pool_height, input_x:input_x + pool_width])
                back_layer_error[b, input_y + unravel_y, input_x + unravel_x] = error[b, out_y, out_x]

            input_x += stride_x
            out_x += 1

        input_y += stride_y
        out_y += 1

    return back_layer_error
