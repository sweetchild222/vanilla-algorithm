import numpy as np

def padding(input, weight_shape):

    (weight_height, weight_width) = weight_shape

    pad_height = weight_height // 2
    pad_width = weight_width // 2

    return np.pad(input, ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant', constant_values=0)


def cal_output_shape(input_shape, weight_shape, stride):

    (input_height, input_width) = input_shape
    (weight_height, weight_width) = weight_shape
    (stride_y, stride_x) = stride

    numerator_height = input_height - weight_height
    numerator_width = input_width - weight_width

    output_shape = ((numerator_height // stride_y) + 1, (numerator_width // stride_x) + 1)

    return output_shape


def padding_output_shape(weight_shape):

    (weight_height, weight_width) = weight_shape

    pad_height = weight_height // 2
    pad_width = weight_width // 2

    return weight_shape + (pad_height, pad_width)


def forward(input, weight, bias, stride):

    input = padding(input, weight.shape)

    (batches, input_height, input_width) = input.shape
    (weight_height, weight_width) = weight.shape
    (stride_y, stride_x) = stride

    output_shape = cal_output_shape(input.shape[-2:], weight.shape, stride)

    output = np.zeros((batches, ) + output_shape)

    input_y = out_y = 0
    while (input_y + weight_height) <= input_height:
        input_x = out_x = 0
        while (input_x + weight_width) <= input_width:

            i = input[:,input_y:input_y + weight_height, input_x:input_x + weight_width]

            iw = i * weight

            iw = iw.reshape((iw.shape[0], -1))
            iw = np.sum(iw, axis=-1)

            output[:, out_y, out_x] =  iw + bias

            input_x += stride_x
            out_x += 1

        input_y += stride_y
        out_y += 1

    return output


def backward(input, error, weight, bias, stride):

    input = padding(input, weight.shape)

    (weight_height, weight_width) = weight.shape
    (batches, input_height, input_width) = input.shape
    (stride_y, stride_x) = stride

    back_layer_error = np.zeros(input.shape)
    batch_weight = np.array([weight] * batches)

    w_delta = np.zeros((batches, ) + weight.shape)
    b_delta = np.zeros((batches, ) + bias.shape)

    input_y = out_y = 0
    while (input_y + weight_height) <= input_height:
        input_x = out_x = 0
        while (input_x + weight_width) <= input_width:
            err = (error[:, out_y, out_x])[:, np.newaxis, np.newaxis]
            i = input[:, input_y:input_y + weight_height, input_x:input_x + weight_width]

            w_delta += (err * i)
            b_delta += err.reshape((batches, ) + bias.shape)

            bw_err = batch_weight * err
            #shallow copy
            bl_err = back_layer_error[:, input_y:input_y + weight_height, input_x:input_x + weight_width]
            bl_err += bw_err

            input_x += stride_x
            out_x += 1


        input_y += stride_y
        out_y += 1

    return back_layer_error, np.sum(w_delta, axis=0), np.sum(b_delta, axis=0)
