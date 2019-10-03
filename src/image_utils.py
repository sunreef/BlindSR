import torch



def reduce_image(img, scale):
    batch, channels, height, width = img.size()
    reduced_img = torch.zeros(batch, channels * scale * scale, height // scale, width // scale).cuda()

    for x in range(scale):
        for y in range(scale):
            for c in range(channels):
                reduced_img[:, c + channels * (y + scale * x), :, :] = img[:, c, x::scale, y::scale]
    return reduced_img

def reconstruct_image(features, scale):
    batch, channels, height, width = features.size()
    img_channels = channels // (scale**2)
    reconstructed_img = torch.zeros(batch, img_channels, height * scale, width * scale).cuda()


    for x in range(scale):
        for y in range(scale):
            for c in range(img_channels):
                f_channel = c + img_channels * (y + scale * x)
                reconstructed_img[:, c, x::scale, y::scale] = features[:, f_channel, :, :]
    return reconstructed_img
