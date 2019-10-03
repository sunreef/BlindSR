import torch

class Degradation:
    def __init__(self, kernel_size, theta=0.0, sigma=[1.0, 1.0]):
        self.kernel_size = kernel_size
        self.theta = torch.tensor([theta])
        self.sigma = torch.tensor(sigma)

        self.kernel = None
        self.build_kernel()

    def cuda(self):
        self.theta = self.theta.cuda()
        self.sigma = self.sigma.cuda()
        self.build_kernel()

    def build_kernel(self):
        kernel_radius = self.kernel_size // 2
        kernel_range = torch.linspace(-kernel_radius, kernel_radius, self.kernel_size)

        if self.sigma.is_cuda:
            kernel_range = kernel_range.cuda()

        horizontal_range = kernel_range[None].repeat((self.kernel_size, 1))
        vertical_range = kernel_range[:, None].repeat((1, self.kernel_size))

        cos_theta = self.theta.cos()
        sin_theta = self.theta.sin()

        cos_theta_2 = cos_theta ** 2
        sin_theta_2 = sin_theta ** 2

        sigma_x_2 = 2.0 * (self.sigma[0] ** 2)
        sigma_y_2 = 2.0 * (self.sigma[1] ** 2)

        a = cos_theta_2 / sigma_x_2 + sin_theta_2 / sigma_y_2
        b = sin_theta * cos_theta * (1.0 / sigma_y_2 - 1.0 / sigma_x_2)
        c = sin_theta_2 / sigma_x_2 + cos_theta_2 / sigma_y_2

        gaussian = lambda x,y: (- ( a * (x ** 2) + 2.0 * b * x * y + c * (y ** 2))).exp()

        kernel = gaussian(horizontal_range, vertical_range)
        kernel /= kernel.sum()
        self.kernel = kernel
        return self.kernel




