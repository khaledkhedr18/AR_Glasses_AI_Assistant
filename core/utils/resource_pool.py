from typing import TypeVar, Generic, List, Optional
from threading import Lock
from core.utils.logging import Logger

T = TypeVar('T')

class ResourcePool(Generic[T]):
    """Generic resource pool for managing shared resources"""

    def __init__(self, max_size: int = 10, resource_type: str = "unknown"):
        self.max_size = max_size
        self.pool: List[T] = []
        self.lock = Lock()
        self.resource_type = resource_type
        self.logger = Logger()

    def get_resource(self) -> Optional[T]:
        """Get a resource from the pool"""
        with self.lock:
            if self.pool:
                resource = self.pool.pop()
                self.logger.debug(f"Retrieved {self.resource_type} from pool")
                return resource
            self.logger.debug(f"No {self.resource_type} available in pool")
            return None

    def return_resource(self, resource: T):
        """Return a resource to the pool"""
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(resource)
                self.logger.debug(f"Returned {self.resource_type} to pool")
            else:
                self.logger.warning(f"Pool full, discarding {self.resource_type}")

    def clear(self):
        """Clear all resources from the pool"""
        with self.lock:
            self.pool.clear()
            self.logger.info(f"Cleared {self.resource_type} pool")

    def size(self) -> int:
        """Get current pool size"""
        with self.lock:
            return len(self.pool)

    def is_full(self) -> bool:
        """Check if pool is full"""
        return self.size() >= self.max_size
