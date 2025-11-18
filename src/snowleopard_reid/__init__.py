"""Snow Leopard Re-Identification Package."""

from snowleopard_reid.images import resize_image_if_needed
from snowleopard_reid.utils import get_device

__all__ = ["resize_image_if_needed", "get_device"]


def main() -> None:
    print("Hello from snowleopard-reid!")
