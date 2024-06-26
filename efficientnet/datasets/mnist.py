import mlconfig
from torch import Tensor
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


class Expand:
    def __call__(self, t: Tensor) -> Tensor:
        return t.expand(3, t.size(1), t.size(2))


@mlconfig.register
class MNISTDataLoader(data.DataLoader):
    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, **kwargs) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                Expand(),
            ]
        )

        dataset = datasets.MNIST(
            root,
            train=train,
            transform=transform,
            download=True,
        )

        super().__init__(dataset=dataset, batch_size=batch_size, **kwargs)
